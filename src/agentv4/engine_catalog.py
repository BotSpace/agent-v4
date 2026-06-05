from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EngineNodeSpec:
    example: dict[str, Any] = field(default_factory=dict)
    required: frozenset[str] = frozenset()
    allowed_branches: frozenset[str] | None = None
    branch_prefixes: tuple[str, ...] = ()


ENGINE_NODE_CATALOG = {
    "triggers": {
        "CommandTriggerNode": {"command": "/start", "withArgs": False, "global": True},
        "MessageTriggerNode": {"type": "text", "filter": "any", "context": {"value": ""}},
        "CallbackQueryTriggerNode": {"filter": "equals", "context": {"value": "callback_data"}},
        "ReplyButtonTriggerNode": {"selectedButton": "Button text"},
        "CronTriggerNode": {"schedule": "0 9 * * *", "enabled": True, "targetChatIds": []},
        "ExternalWebhookTriggerNode": {},
    },
    "messages_media": {
        "SendTextMessageNode": {"messageText": "Text", "parseMode": "HTML"},
        "EditOrSendMessageNode": {"messageText": "Text", "parseMode": "HTML", "keyboard": {"active": "none"}},
        "EditMessageNode": {"newText": "Updated text", "parseMode": "HTML"},
        "DeleteMessageNode": {"deleteFromContext": True},
        "ForwardMessageNode": {"fromChatId": 123, "messageId": 456},
        "CopyMessageNode": {"fromChatId": 123, "messageId": 456},
        "PinMessageNode": {"messageId": 456},
        "UnpinMessageNode": {"messageId": 456},
        "UnpinAllMessagesNode": {},
        "SendPhotoNode": {"photoSourceType": "url", "photoUrl": "https://example.com/photo.jpg", "caption": ""},
        "SendVideoNode": {"videoSourceType": "url", "videoUrl": "https://example.com/video.mp4", "caption": ""},
        "SendAudioNode": {"audioSourceType": "url", "audioUrl": "https://example.com/audio.mp3", "caption": ""},
        "SendFileNode": {"fileSourceType": "url", "fileUrl": "https://example.com/file.pdf", "caption": ""},
        "SendAnimationNode": {"animationSourceType": "url", "animationUrl": "https://example.com/animation.gif"},
        "SendVoiceNode": {"voiceSourceType": "url", "voiceUrl": "https://example.com/voice.ogg"},
        "SendVideoNoteNode": {"videoNoteSourceType": "url", "videoNoteUrl": "https://example.com/video-note.mp4"},
        "SendStickerNode": {"stickerSourceType": "file_id", "stickerFileId": "sticker_file_id"},
        "SendMediaGroupNode": {"mediaItems": []},
        "SendLocationNode": {"latitude": 41.3111, "longitude": 69.2797},
        "SendVenueNode": {"latitude": 41.3111, "longitude": 69.2797, "title": "Venue", "address": "Address"},
        "SendContactNode": {"phoneNumber": "+998901234567", "firstName": "Name"},
        "SendPollNode": {"question": "Question?", "options": ["A", "B"], "type": "regular"},
        "SendDiceNode": {"emoji": "🎲"},
        "ChatActionNode": {"action": "typing"},
        "CallbackQueryAnswerNode": {"text": "OK", "showAlert": False},
        "AnswerInlineQueryNode": {"results": []},
    },
    "flow_control": {
        "IfConditionNode": {
            "operator": "AND",
            "conditions": [{"variable": "state_age", "operator": ">=", "value": 18}],
        },
        "ElseIfNode": {"conditions": [{"variable": "state_value", "operator": "equals", "value": "x"}]},
        "ElseNode": {},
        "DelayNode": {"delaySeconds": 1},
        "RandomNode": {"options": ["A", "B"]},
        "ForLoopNode": {"loopMode": "array", "dataKey": "items", "itemVariable": "item", "loopID": "main_loop"},
        "ForLoopContinueNode": {"loopID": "main_loop"},
        "SchedulerNode": {"targetNodeId": "send_reminder", "delaySeconds": 3600},
        "CronNode": {"schedule": "0 9 * * *", "enabled": True},
        "CheckMembershipNode": {"channels": [{"channelUsername": "@channel"}]},
    },
    "data_integrations": {
        "HTTPRequestNode": {
            "method": "GET",
            "url": "https://api.example.com/data",
            "headers": {},
            "responseVariable": "api_response",
            "timeout": 10,
        },
        "CustomCodeNode": {"jsCode": "context.result = true;", "timeout": 5},
        "StateNode": {"key": "name", "value": "{{state_name}}", "type": "text"},
        "VariableNode": {"variableName": "counter", "operation": "set", "value": 1},
        "CollectionNode": {"collection_name": "leads", "fieldMappings": {}},
        "LoadCollectionItemNode": {"collection": {"name": "leads"}, "contextKey": "lead", "filters": []},
        "LoadCollectionListNode": {"collection": {"name": "leads"}, "contextKey": "leads", "filters": [], "limit": 100},
        "UpdateCollectionNode": {"collection_name": "leads", "filters": [], "fieldMappings": {}},
        "DeleteCollectionNode": {"collection_name": "leads", "filters": []},
        "DownloadNode": {"file": "{{state_file_id}}", "key": "downloaded_file"},
        "SendToAdminNode": {"message": "New request", "adminChatIds": []},
        "PaymentNode": {
            "paymentType": "click",
            "amount": 10000,
            "comment": "To'lov",
            "buttonText": "💳 To'lash",
        },
    },
}


ENGINE_NODE_SPECS: dict[str, EngineNodeSpec] = {
    # Triggers
    "CommandTriggerNode": EngineNodeSpec(ENGINE_NODE_CATALOG["triggers"]["CommandTriggerNode"], frozenset({"command"})),
    "MessageTriggerNode": EngineNodeSpec(ENGINE_NODE_CATALOG["triggers"]["MessageTriggerNode"], frozenset({"type"})),
    "CallbackQueryTriggerNode": EngineNodeSpec(ENGINE_NODE_CATALOG["triggers"]["CallbackQueryTriggerNode"]),
    "ReplyButtonTriggerNode": EngineNodeSpec(ENGINE_NODE_CATALOG["triggers"]["ReplyButtonTriggerNode"]),
    "CronTriggerNode": EngineNodeSpec(
        ENGINE_NODE_CATALOG["triggers"]["CronTriggerNode"],
        frozenset({"schedule", "enabled", "targetChatIds"}),
    ),
    "ExternalWebhookTriggerNode": EngineNodeSpec(ENGINE_NODE_CATALOG["triggers"]["ExternalWebhookTriggerNode"]),
    # Messages and Telegram operations
    "SendTextMessageNode": EngineNodeSpec(
        ENGINE_NODE_CATALOG["messages_media"]["SendTextMessageNode"], frozenset({"messageText"})
    ),
    "EditOrSendMessageNode": EngineNodeSpec(
        ENGINE_NODE_CATALOG["messages_media"]["EditOrSendMessageNode"], frozenset({"messageText"})
    ),
    "EditMessageNode": EngineNodeSpec(ENGINE_NODE_CATALOG["messages_media"]["EditMessageNode"], frozenset({"newText"})),
    "DeleteMessageNode": EngineNodeSpec(ENGINE_NODE_CATALOG["messages_media"]["DeleteMessageNode"]),
    "ForwardMessageNode": EngineNodeSpec(
        ENGINE_NODE_CATALOG["messages_media"]["ForwardMessageNode"], frozenset({"fromChatId", "messageId"})
    ),
    "CopyMessageNode": EngineNodeSpec(
        ENGINE_NODE_CATALOG["messages_media"]["CopyMessageNode"], frozenset({"fromChatId", "messageId"})
    ),
    "PinMessageNode": EngineNodeSpec(ENGINE_NODE_CATALOG["messages_media"]["PinMessageNode"], frozenset({"messageId"})),
    "UnpinMessageNode": EngineNodeSpec(
        ENGINE_NODE_CATALOG["messages_media"]["UnpinMessageNode"], frozenset({"messageId"})
    ),
    "UnpinAllMessagesNode": EngineNodeSpec(ENGINE_NODE_CATALOG["messages_media"]["UnpinAllMessagesNode"]),
    "SendPhotoNode": EngineNodeSpec(
        ENGINE_NODE_CATALOG["messages_media"]["SendPhotoNode"], frozenset({"photoSourceType"})
    ),
    "SendVideoNode": EngineNodeSpec(
        ENGINE_NODE_CATALOG["messages_media"]["SendVideoNode"], frozenset({"videoSourceType"})
    ),
    "SendAudioNode": EngineNodeSpec(
        ENGINE_NODE_CATALOG["messages_media"]["SendAudioNode"], frozenset({"audioSourceType"})
    ),
    "SendFileNode": EngineNodeSpec(ENGINE_NODE_CATALOG["messages_media"]["SendFileNode"], frozenset({"fileSourceType"})),
    "SendAnimationNode": EngineNodeSpec(
        ENGINE_NODE_CATALOG["messages_media"]["SendAnimationNode"], frozenset({"animationSourceType"})
    ),
    "SendVoiceNode": EngineNodeSpec(
        ENGINE_NODE_CATALOG["messages_media"]["SendVoiceNode"], frozenset({"voiceSourceType"})
    ),
    "SendVideoNoteNode": EngineNodeSpec(
        ENGINE_NODE_CATALOG["messages_media"]["SendVideoNoteNode"], frozenset({"videoNoteSourceType"})
    ),
    "SendStickerNode": EngineNodeSpec(
        ENGINE_NODE_CATALOG["messages_media"]["SendStickerNode"], frozenset({"stickerSourceType"})
    ),
    "SendMediaGroupNode": EngineNodeSpec(
        ENGINE_NODE_CATALOG["messages_media"]["SendMediaGroupNode"], frozenset({"mediaItems"})
    ),
    "SendLocationNode": EngineNodeSpec(
        ENGINE_NODE_CATALOG["messages_media"]["SendLocationNode"], frozenset({"latitude", "longitude"})
    ),
    "SendVenueNode": EngineNodeSpec(
        ENGINE_NODE_CATALOG["messages_media"]["SendVenueNode"], frozenset({"latitude", "longitude", "title", "address"})
    ),
    "SendContactNode": EngineNodeSpec(
        ENGINE_NODE_CATALOG["messages_media"]["SendContactNode"], frozenset({"phoneNumber", "firstName"})
    ),
    "SendPollNode": EngineNodeSpec(
        ENGINE_NODE_CATALOG["messages_media"]["SendPollNode"], frozenset({"question", "options"})
    ),
    "SendDiceNode": EngineNodeSpec(ENGINE_NODE_CATALOG["messages_media"]["SendDiceNode"]),
    "ChatActionNode": EngineNodeSpec(ENGINE_NODE_CATALOG["messages_media"]["ChatActionNode"], frozenset({"action"})),
    "CallbackQueryAnswerNode": EngineNodeSpec(ENGINE_NODE_CATALOG["messages_media"]["CallbackQueryAnswerNode"]),
    "AnswerInlineQueryNode": EngineNodeSpec(ENGINE_NODE_CATALOG["messages_media"]["AnswerInlineQueryNode"]),
    # Flow control
    "IfConditionNode": EngineNodeSpec(
        ENGINE_NODE_CATALOG["flow_control"]["IfConditionNode"],
        frozenset({"conditions"}),
        frozenset({"true", "false", "next"}),
        ("branch_",),
    ),
    "ElseIfNode": EngineNodeSpec(
        ENGINE_NODE_CATALOG["flow_control"]["ElseIfNode"], frozenset({"conditions"}), frozenset({"true", "next"})
    ),
    "ElseNode": EngineNodeSpec(ENGINE_NODE_CATALOG["flow_control"]["ElseNode"], allowed_branches=frozenset({"true"})),
    "DelayNode": EngineNodeSpec(ENGINE_NODE_CATALOG["flow_control"]["DelayNode"]),
    "RandomNode": EngineNodeSpec(ENGINE_NODE_CATALOG["flow_control"]["RandomNode"], frozenset({"options"}), branch_prefixes=("option_",)),
    "ForLoopNode": EngineNodeSpec(
        ENGINE_NODE_CATALOG["flow_control"]["ForLoopNode"],
        allowed_branches=frozenset({"loop-body", "no-items"}),
    ),
    "ForLoopContinueNode": EngineNodeSpec(
        ENGINE_NODE_CATALOG["flow_control"]["ForLoopContinueNode"],
        allowed_branches=frozenset({"loop-continue", "loop-done"}),
    ),
    "SchedulerNode": EngineNodeSpec(
        ENGINE_NODE_CATALOG["flow_control"]["SchedulerNode"], frozenset({"targetNodeId"})
    ),
    "CronNode": EngineNodeSpec(ENGINE_NODE_CATALOG["flow_control"]["CronNode"], frozenset({"schedule", "enabled"})),
    "CheckMembershipNode": EngineNodeSpec(
        ENGINE_NODE_CATALOG["flow_control"]["CheckMembershipNode"],
        allowed_branches=frozenset({"is-member", "not-member"}),
    ),
    # Data and integrations
    "HTTPRequestNode": EngineNodeSpec(
        ENGINE_NODE_CATALOG["data_integrations"]["HTTPRequestNode"], frozenset({"method", "url"})
    ),
    "CustomCodeNode": EngineNodeSpec(ENGINE_NODE_CATALOG["data_integrations"]["CustomCodeNode"]),
    "StateNode": EngineNodeSpec(ENGINE_NODE_CATALOG["data_integrations"]["StateNode"], frozenset({"key", "type"})),
    "VariableNode": EngineNodeSpec(
        ENGINE_NODE_CATALOG["data_integrations"]["VariableNode"], frozenset({"variableName", "operation"})
    ),
    "CollectionNode": EngineNodeSpec(
        ENGINE_NODE_CATALOG["data_integrations"]["CollectionNode"], frozenset({"collection_name", "fieldMappings"})
    ),
    "LoadCollectionItemNode": EngineNodeSpec(
        ENGINE_NODE_CATALOG["data_integrations"]["LoadCollectionItemNode"],
        frozenset({"collection", "contextKey"}),
        frozenset({"found", "not_found"}),
    ),
    "LoadCollectionListNode": EngineNodeSpec(
        ENGINE_NODE_CATALOG["data_integrations"]["LoadCollectionListNode"], frozenset({"collection", "contextKey"})
    ),
    "UpdateCollectionNode": EngineNodeSpec(
        ENGINE_NODE_CATALOG["data_integrations"]["UpdateCollectionNode"], frozenset({"collection_name", "fieldMappings"})
    ),
    "DeleteCollectionNode": EngineNodeSpec(
        ENGINE_NODE_CATALOG["data_integrations"]["DeleteCollectionNode"], frozenset({"collection_name"})
    ),
    "DownloadNode": EngineNodeSpec(ENGINE_NODE_CATALOG["data_integrations"]["DownloadNode"], frozenset({"file", "key"})),
    "SendToAdminNode": EngineNodeSpec(ENGINE_NODE_CATALOG["data_integrations"]["SendToAdminNode"]),
    # Botspace orqali Click to'lovi — paid/failed branch. Egasi to'lov tizimi bilan integratsiya qilmaydi.
    "PaymentNode": EngineNodeSpec(
        ENGINE_NODE_CATALOG["data_integrations"]["PaymentNode"],
        frozenset({"amount"}),
        frozenset({"paid", "failed"}),
    ),
}


ENGINE_NODE_TYPES = frozenset(ENGINE_NODE_SPECS)


def get_engine_node_spec(node_type: str) -> EngineNodeSpec | None:
    return ENGINE_NODE_SPECS.get(node_type)


def is_valid_engine_branch(node_type: str, branch: str | None) -> bool:
    if branch is None:
        return True
    spec = get_engine_node_spec(node_type)
    if spec is None:
        return False
    if spec.allowed_branches is None and not spec.branch_prefixes:
        return True
    if spec.allowed_branches is not None and branch in spec.allowed_branches:
        return True
    return any(branch.startswith(prefix) for prefix in spec.branch_prefixes)


def engine_node_requires_branch(node_type: str) -> bool:
    spec = get_engine_node_spec(node_type)
    return spec is not None and (spec.allowed_branches is not None or bool(spec.branch_prefixes))


def validate_engine_node_data(node_type: str, data: dict[str, Any]) -> list[str]:
    spec = get_engine_node_spec(node_type)
    if spec is None:
        return [f"Unknown engine node type `{node_type}`."]
    missing = sorted(
        key
        for key in spec.required
        if key not in data or data[key] is None or data[key] == ""
    )
    return [f"`{node_type}` requires data.{key}." for key in missing]


def engine_catalog_prompt() -> str:
    lines = [
        "Engine-golang node catalog for add_engine_node(node_type, data):",
        "Use high-level commands for common text/buttons/questions. Use add_engine_node when the requested bot needs engine features below.",
    ]
    for group, nodes in ENGINE_NODE_CATALOG.items():
        lines.append(f"{group}:")
        for node_type, example in nodes.items():
            spec = ENGINE_NODE_SPECS[node_type]
            required = f"; required={sorted(spec.required)}" if spec.required else ""
            branches = ""
            if spec.allowed_branches is not None:
                branches = f"; branch_handles={sorted(spec.allowed_branches)}"
            elif spec.branch_prefixes:
                branches = f"; branch_prefixes={list(spec.branch_prefixes)}"
            lines.append(f"- {node_type}: data example {example}{required}{branches}")
    return "\n".join(lines)
