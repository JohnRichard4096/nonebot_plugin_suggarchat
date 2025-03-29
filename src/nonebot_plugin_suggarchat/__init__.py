from nonebot.plugin import PluginMetadata, require

from . import API, config, connection, event, resources, suggar

require("nonebot_plugin_localstore")


__all__ = [
    "API",
    "config",
    "connection",
    "event",
    "resources",
    "suggar",
]

__plugin_meta__ = PluginMetadata(
    name="SuggarChat 高可扩展性大模型聊天插件/框架",
    description="强大的聊天插件/框架，内建OpenAI协议客户端实现，高可扩展性，多模型切换，事件API提供，完全的上下文支持，适配Nonebot2-Onebot-V11适配器",
    usage="https://github.com/JohnRichard4096/nonebot_plugin_suggarchat/wiki",
    homepage="https://github.com/JohnRichard4096/nonebot_plugin_suggarchat/",
    type="application",
    supported_adapters={"~onebot.v11"},
)
