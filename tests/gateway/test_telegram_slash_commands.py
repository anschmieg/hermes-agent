"""Tests for Telegram slash command routing via _handle_text_message.

Before the fix, Telegram used a separate CommandHandler that intercepted
"/command" messages at the Bot API level and routed them to _handle_command,
which bypassed text batching and could result in commands not being properly
dispatched to the gateway's command handler.

After the fix, all text (including slash commands) flows through
_handle_text_message, which:
  1. Detects leading "/" and passes is_command=True to _should_process_message
     (ensuring group-chat mention requirements are bypassed for commands)
  2. Sets MessageType.COMMAND on the event
  3. Routes through _enqueue_text_event -> handle_message -> gateway dispatcher
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType, SessionSource


def _make_update(text: str, chat_id: int = 12345, chat_type: str = "private"):
    """Build a minimal telegram.Update-like object with the attributes we need."""
    user = SimpleNamespace(id=1, full_name="Test User", username="testuser")
    chat = SimpleNamespace(
        id=chat_id,
        type=chat_type,
        title=None,
        full_name="Test Chat",
    )
    return SimpleNamespace(
        message=SimpleNamespace(
            message_id=1,
            text=text,
            date=None,
            chat=chat,
            from_user=user,
            message_thread_id=None,
        )
    )


def _make_adapter():
    """Create a minimal TelegramAdapter for testing."""
    from gateway.platforms.telegram import TelegramAdapter

    config = PlatformConfig(enabled=True, token="test-token", extra={})
    adapter = object.__new__(TelegramAdapter)
    adapter._platform = Platform.TELEGRAM
    adapter.config = config
    adapter._pending_text_batches = {}
    adapter._pending_text_batch_tasks = {}
    adapter._text_batch_delay_seconds = 0.05  # fast for tests
    adapter._active_sessions = {}
    adapter._pending_messages = {}
    adapter._message_handler = AsyncMock()
    adapter.handle_message = AsyncMock()
    adapter._should_process_message = MagicMock(return_value=True)
    adapter._clean_bot_trigger_text = MagicMock(side_effect=lambda x: x)
    # Mock _build_message_event to return a real MessageEvent (avoids telegram internals)
    def real_build(src_msg, src_type):
        return MessageEvent(
            text=src_msg.text,
            message_type=src_type,
            source=SessionSource(
                platform=Platform.TELEGRAM,
                chat_id=str(src_msg.chat.id),
                chat_type="dm",
            ),
        )
    adapter._build_message_event = MagicMock(side_effect=real_build)
    adapter._enqueue_text_event = MagicMock()
    return adapter


class TestSlashCommandRouting:
    """Verify slash commands are properly detected and routed."""

    @pytest.mark.asyncio
    async def test_slash_command_calls_build_message_event_with_command_type(self):
        """A leading '/' should cause _build_message_event to be called with MessageType.COMMAND."""
        adapter = _make_adapter()

        update = _make_update("/help")
        await adapter._handle_text_message(update, MagicMock())

        adapter._build_message_event.assert_called_once()
        call_args = adapter._build_message_event.call_args
        _, msg_type = call_args[0]
        assert msg_type == MessageType.COMMAND, f"Expected COMMAND, got {msg_type}"

    @pytest.mark.asyncio
    async def test_regular_text_calls_build_message_event_with_text_type(self):
        """Regular text (no leading '/') should cause _build_message_event to be called with MessageType.TEXT."""
        adapter = _make_adapter()

        update = _make_update("hello world")
        await adapter._handle_text_message(update, MagicMock())

        adapter._build_message_event.assert_called_once()
        call_args = adapter._build_message_event.call_args
        _, msg_type = call_args[0]
        assert msg_type == MessageType.TEXT, f"Expected TEXT, got {msg_type}"

    @pytest.mark.asyncio
    async def test_slash_command_passes_is_command_true(self):
        """'/skills' should call _should_process_message with is_command=True."""
        adapter = _make_adapter()

        update = _make_update("/skills python")
        await adapter._handle_text_message(update, MagicMock())

        adapter._should_process_message.assert_called_once()
        call_kwargs = adapter._should_process_message.call_args.kwargs
        assert call_kwargs.get("is_command") is True, (
            f"Expected is_command=True, got {call_kwargs}"
        )

    @pytest.mark.asyncio
    async def test_regular_text_passes_is_command_false(self):
        """'hello world' should call _should_process_message with is_command=False."""
        adapter = _make_adapter()

        update = _make_update("hello world")
        await adapter._handle_text_message(update, MagicMock())

        adapter._should_process_message.assert_called_once()
        call_kwargs = adapter._should_process_message.call_args.kwargs
        assert call_kwargs.get("is_command") is False, (
            f"Expected is_command=False, got {call_kwargs}"
        )

    @pytest.mark.asyncio
    async def test_slash_command_enqueues_event_with_command_type(self):
        """A slash command should be enqueued via _enqueue_text_event with MessageType.COMMAND."""
        adapter = _make_adapter()

        update = _make_update("/status")
        await adapter._handle_text_message(update, MagicMock())

        adapter._enqueue_text_event.assert_called_once()
        enqueued_event = adapter._enqueue_text_event.call_args[0][0]
        assert enqueued_event.message_type == MessageType.COMMAND

    @pytest.mark.asyncio
    async def test_regular_text_enqueues_event_with_text_type(self):
        """Regular text should be enqueued via _enqueue_text_event with MessageType.TEXT."""
        adapter = _make_adapter()

        update = _make_update("hello")
        await adapter._handle_text_message(update, MagicMock())

        adapter._enqueue_text_event.assert_called_once()
        enqueued_event = adapter._enqueue_text_event.call_args[0][0]
        assert enqueued_event.message_type == MessageType.TEXT

    @pytest.mark.asyncio
    async def test_slash_command_bypasses_mention_requirement_in_groups(self):
        """In groups, '/' commands should pass is_command=True to bypass mention requirement."""
        adapter = _make_adapter()

        def group_should_process(msg, *, is_command=False):
            return is_command  # reject non-commands

        adapter._should_process_message = MagicMock(side_effect=group_should_process)

        update = _make_update("/status", chat_type="group")
        await adapter._handle_text_message(update, MagicMock())

        # If is_command=False, _should_process_message would return False
        # and the message would be rejected (early return). Since we got here,
        # is_command must have been True.
        # But let's also verify by checking the call
        call_kwargs = adapter._should_process_message.call_args.kwargs
        assert call_kwargs.get("is_command") is True, (
            "is_command should be True to bypass group mention requirement"
        )


class TestCleanBotTriggerText:
    """Verify _clean_bot_trigger_text handles command text correctly."""

    def test_clean_removes_username_suffix(self):
        """Bot username suffix like '@mybot' should be stripped from command text."""
        from gateway.platforms.telegram import TelegramAdapter

        config = PlatformConfig(enabled=True, token="test-token", extra={})
        adapter = object.__new__(TelegramAdapter)
        adapter._bot = SimpleNamespace(username="mybot")

        result = adapter._clean_bot_trigger_text("/help @mybot")
        assert result == "/help"

    def test_clean_preserves_command_args(self):
        """Args after the command name should be preserved."""
        from gateway.platforms.telegram import TelegramAdapter

        config = PlatformConfig(enabled=True, token="test-token", extra={})
        adapter = object.__new__(TelegramAdapter)
        adapter._bot = SimpleNamespace(username="mybot")

        result = adapter._clean_bot_trigger_text("/plan build rocket @mybot extra")
        assert result == "/plan build rocket extra"

    def test_clean_handles_no_username(self):
        """Without a bot username set, text should be returned unchanged."""
        from gateway.platforms.telegram import TelegramAdapter

        config = PlatformConfig(enabled=True, token="test-token", extra={})
        adapter = object.__new__(TelegramAdapter)
        adapter._bot = None

        result = adapter._clean_bot_trigger_text("/help")
        assert result == "/help"

class TestNoCommandHandlerRegistered:
    """Verify the CommandHandler is no longer registered."""

    def test_no_command_handler_in_connect_source(self):
        """connect() should not register a CommandHandler.

        All text including slash commands should flow through the
        MessageHandler with filters.TEXT (not filters.COMMAND).
        """
        import inspect
        from gateway.platforms.telegram import TelegramAdapter

        source = inspect.getsource(TelegramAdapter.connect)
        # The word "CommandHandler" must not appear as a handler registration
        assert "CommandHandler" not in source, (
            "CommandHandler must not appear in connect() — "
            "use MessageHandler(filters.TEXT) for all text including commands"
        )

