import asyncio
import contextlib
import random
import sys
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import nonebot
import openai
from nonebot import logger, on_command, on_message, on_notice
from nonebot.adapters import Bot, Message
from nonebot.adapters.onebot.v11 import (
    MessageSegment,
)
from nonebot.adapters.onebot.v11.event import (
    GroupIncreaseNoticeEvent,
    GroupMessageEvent,
    GroupRecallNoticeEvent,
    MessageEvent,
    PokeNotifyEvent,
    PrivateMessageEvent,
    Reply,
)
from nonebot.exception import NoneBotException
from nonebot.matcher import Matcher
from nonebot.params import CommandArg
from nonebot.rule import to_me
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

from .config import Config, config_manager
from .event import ChatEvent, EventType, PokeEvent
from .matcher import SuggarMatcher
from .resources import (
    get_current_datetime_timestamp,
    get_friend_info,
    get_memory_data,
    hybrid_token_count,
    split_message_into_chats,
    synthesize_message,
    write_memory_data,
)


@dataclass
class ChatManager:
    debug: bool = False
    session_clear_group: list[dict[str, Any]] = field(default_factory=list)
    session_clear_user: list[dict[str, Any]] = field(default_factory=list)
    custom_menu: list[dict[str, str]] = field(default_factory=list)
    running_messages_poke: dict[str, Any] = field(default_factory=dict)
    menu_msg: str = "聊天功能菜单:\n/聊天菜单 唤出菜单 \n/del_memory 丢失这个群/聊天的记忆 \n/enable 在群聊启用聊天 \n/disable 在群聊里关闭聊天\n/prompt <arg> [text] 设置聊群自定义补充prompt（--(show) 展示当前提示词，--(clear) 清空当前prompt，--(set) [文字]则设置提示词，e.g.:/prompt --(show)）,/prompt --(set) [text]。）\n/sessions指令帮助：\nset：覆盖当前会话为指定编号的会话\ndel：删除指定编号的会话\narchive：归档当前会话\nclear：清空所有会话\nPreset帮助：\n/presets 列出所有读取到的模型预设\n/set_preset 或 /设置预设 或 /设置模型预设  <预设名> 设置当前使用的预设"


chat_manager = ChatManager()


async def openai_get_chat(
    base_url: str,
    model: str,
    key: str,
    messages: list,
    max_tokens: int,
    config: Config,
    bot: Bot,
) -> str:
    if (
        not str(config.open_ai_base_url).strip()
        or not str(config.open_ai_api_key).strip()
    ):
        raise RuntimeError("错误！OpenAI Url或Key为空！")
    client = openai.AsyncOpenAI(
        base_url=base_url, api_key=key, timeout=config.llm_timeout
    )
    # 创建聊天完成请求
    for index, i in enumerate(range(3)):
        try:
            completion: (
                ChatCompletion | openai.AsyncStream[ChatCompletionChunk]
            ) = await client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                stream=config.stream,
            )
            break
        except Exception as e:
            logger.error(f"发生了错误: {e}")
            logger.info(f"尝试第{i + 1}次重试")
            if index == 2:
                logger.error("获取对话失败，请检查你的API Key和API base_url是否正确！")
                await send_to_admin(f"在获取对话时发生了错误: {e}", bot)
                raise e
            continue

    response = ""
    if config.stream and isinstance(completion, openai.AsyncStream):
        # 流式接收响应并构建最终的聊天文本
        async for chunk in completion:
            try:
                if chunk.choices[0].delta.content is not None:
                    response += chunk.choices[0].delta.content
                    if chat_manager.debug:
                        logger.debug(chunk.choices[0].delta.content)
            except IndexError:
                break
        # 记录生成的响应日志
    else:
        if chat_manager.debug:
            logger.debug(response)
        if isinstance(completion, ChatCompletion):
            response = completion.choices[0].message.content
        else:
            raise RuntimeError("Unexpected completion type received.")
    return response or ""


protocols_adapters: dict[
    str, Callable[[str, str, str, list, int, Config, Bot], Coroutine[Any, Any, str]]
] = {"openai-builtin": openai_get_chat}


async def send_to_admin(msg: str, bot: Bot | None = None) -> None:
    """
    异步发送消息给管理员。

    该函数会检查配置文件是否允许发送消息给管理员，以及是否配置了管理员群号。
    如果满足条件，则发送消息；否则，将记录警告日志。

    参数:
    msg (str): 要发送给管理员的消息。

    返回:
    无返回值。
    """
    # 检查是否允许发送消息给管理员
    if not config_manager.config.allow_send_to_admin:
        return
    # 检查管理员群号是否已配置
    if config_manager.config.admin_group == 0:
        try:
            # 如果未配置管理员群号但尝试发送消息，抛出警告
            raise RuntimeWarning("错误！管理员群组没有被设定！")
        except Exception:
            # 记录警告日志并捕获异常信息
            logger.warning(f'Admin 群组没有被设定，"{msg}"不会被推送！')
            exc_type, exc_vaule, exc_tb = sys.exc_info()
            logger.exception(f"{exc_type}:{exc_vaule}")
        return
    # 获取bot实例并发送消息到管理员群
    if bot:
        await bot.send_group_msg(
            group_id=config_manager.config.admin_group, message=msg
        )
    else:
        await (nonebot.get_bot()).send_group_msg(
            group_id=config_manager.config.admin_group, message=msg
        )


# fakepeople rule
async def rule(event: MessageEvent, bot: Bot) -> bool:
    """
    根据配置和消息事件判断是否触发回复的规则。

    参数:
    - event: MessageEvent 类型的事件，包含消息事件的详细信息。

    - bot: Bot 类型的机器人实例，用于调用机器人相关方法。

    返回值:
    - bool 类型，表示是否触发回复的规则。
    """
    # 获取消息内容并去除前后空格
    message = event.get_message()
    message_text = message.extract_plain_text().strip()

    # 如果不是群消息事件，则总是返回 True，表示总是回复私聊消息
    if not isinstance(event, GroupMessageEvent):
        return True

    # 根据配置中的 keyword 判断是否需要回复
    if config_manager.config.keyword == "at":
        # 如果配置中的 keyword 为 "at"，则当消息是提到机器人时回复
        if event.is_tome():
            return True
    # 如果配置中的 keyword 不为 "at"，则当消息文本以 keyword 开头时回复
    elif message_text.startswith(config_manager.config.keyword):
        """开头为{keyword}必定回复"""
        return True

    # 如果没有开启随机回复功能，则不回复
    if config_manager.config.fake_people:
        # 私聊消息不进行随机回复
        if event.is_tome() and isinstance(event, PrivateMessageEvent):
            """私聊过滤"""
            return False

        # 根据随机率判断是否回复
        rand = random.random()
        rate = config_manager.config.probability
        if rand <= rate:
            return True

        # 获取记忆数据
        memory_data: dict = get_memory_data(event)

        # 合成消息内容
        content = await synthesize_message(message, bot)

        # 获取当前时间戳
        Date = get_current_datetime_timestamp()

        # 获取消息发送者的角色（群成员或管理员等）
        role = (
            await bot.get_group_member_info(
                group_id=event.group_id, user_id=event.user_id
            )
        )["role"]
        if role == "admin":
            role = "群管理员"
        elif role == "owner":
            role = "群主"
        elif role == "member":
            role = "普通成员"

        # 获取消息发送者的用户ID和昵称
        user_id = event.user_id
        user_name = (
            await bot.get_group_member_info(
                group_id=event.group_id, user_id=event.user_id
            )
        )["nickname"]

        # 构造消息记录格式
        content_message = f"[{role}][{Date}][{user_name}（{user_id}）]说:{content}"

        # 将消息记录添加到记忆数据中
        memory_data["memory"]["messages"].append(
            {"role": "user", "content": content_message}
        )

        # 更新记忆数据
        write_memory_data(event, memory_data)

    return False


async def is_member(event: GroupMessageEvent, bot: Bot) -> bool:
    """
    判断事件触发者是否为群组普通成员。

    本函数通过调用机器人API获取事件触发者在群组中的角色信息，以确定其是否为普通成员。

    参数:
    - event: GroupMessageEvent - 群组消息事件，包含事件相关数据如群组ID和用户ID。
    - bot: Bot - 机器人实例，用于调用API获取群组成员信息。

    返回:
    - bool: 如果事件触发者是群组普通成员，则返回True，否则返回False。
    """
    # 获取群组成员信息
    user_role = await bot.get_group_member_info(
        group_id=event.group_id, user_id=event.user_id
    )
    # 提取成员在群组中的角色
    user_role = user_role.get("role")
    # 判断成员角色是否为"member"（普通成员）
    return user_role == "member"


async def get_chat(
    messages: list,
    bot: Bot | None = None,
) -> str:
    """
    异步获取聊天响应函数

    本函数根据输入的消息列表生成聊天响应文本它根据配置文件中的设置，
    选择适当的API密钥、基础URL和模型进行聊天生成支持流式生成响应文本，
    并处理配置中预设的加载和错误情况下的配置重置

    参数:
    - messages (list): 包含聊天消息的列表，每个消息是一个字典

    返回:
    - str: 生成的聊天响应文本
    """

    # 声明全局变量，用于访问配置和判断是否启用
    # 从配置中获取最大token数量
    max_tokens = config_manager.config.max_tokens
    func = openai_get_chat
    # 根据配置中的预设值，选择不同的API密钥和基础URL
    if config_manager.config.preset == "__main__":
        # 如果是主配置，直接使用配置文件中的设置
        base_url = config_manager.config.open_ai_base_url
        key = config_manager.config.open_ai_api_key
        model = config_manager.config.model
        protocol = config_manager.config.protocol
    else:
        # 如果是其他预设，从模型列表中查找匹配的设置
        for i in config_manager.get_models():
            if i.name == config_manager.config.preset:
                base_url = i.base_url
                key = i.api_key
                model = i.model
                protocol = i.protocol
                break
        else:
            # 如果未找到匹配的预设，记录错误并重置预设为主配置文件
            logger.error(
                f"预设 {config_manager.config.preset} 未找到，已重置为主配置文件"
            )
            logger.info(f"找到：模型：{config_manager.config.model}")
            config_manager.config.preset = "__main__"
            key = config_manager.config.open_ai_api_key
            model = config_manager.config.model
            base_url = config_manager.config.open_ai_base_url
            protocol = config_manager.config.protocol
            # 保存更新后的配置
            config_manager.save_config()
    if protocol == "__main__":
        func = openai_get_chat

    elif protocol not in protocols_adapters:
        raise Exception(f"协议 {protocol} 的适配器未找到!")
    else:
        func = protocols_adapters[protocol]
    # 记录日志，开始获取对话
    logger.debug(f"开始获取 {model} 的对话")
    logger.debug(f"预设：{config_manager.config.preset}")
    logger.debug(f"密钥：{key[:7]}...")
    logger.debug(f"协议：{protocol}")
    logger.debug(f"API地址：{base_url}")
    return await func(
        base_url,
        model,
        key,
        messages,
        max_tokens,
        config_manager.config,
        bot or nonebot.get_bot(),
    )


# 创建响应器实例
menu = on_command("聊天菜单", block=True, aliases={"chat_menu"}, priority=10)
del_memory = on_command(
    "del_memory",
    aliases={"失忆", "删除记忆", "删除历史消息", "删除回忆"},
    block=True,
    priority=10,
)
enable = on_command("enable_chat", aliases={"启用聊天"}, block=True, priority=10)
disable = on_command("disable_chat", aliases={"禁用聊天"}, block=True, priority=10)
prompt = on_command("prompt", priority=10, block=True)
presets = on_command("presets", priority=10, block=True)
set_preset = on_command(
    "set_preset", aliases={"设置预设", "设置模型预设"}, priority=10, block=True
)
# del_all_memory = on_command("del_all_memory",priority=10,block=True)
sessions = on_command("sessions", priority=10, block=True)
debug_switch = on_command("debug", priority=10, block=True)
add_notice = on_notice(block=False)
poke = on_notice(priority=10, block=True)
chat = on_message(
    block=False, priority=11, rule=rule
)  # 不再在此处判断是否触发,转到rule方法
debug_handle = on_message(rule=to_me(), priority=10, block=False)
recall = on_notice()
# 可选择prompt响应器
choose_prompt = on_command("choose_prompt", priority=10, block=True)


@choose_prompt.handle()
async def _(event: MessageEvent, args: Message = CommandArg()):
    """处理选择提示词的命令"""

    async def display_current_prompts() -> None:
        """显示当前群组和私聊的提示词预设"""
        msg = (
            f"当前群组的提示词预设：{config_manager.config.group_prompt_character}\n"
            f"当前私聊的提示词预设：{config_manager.config.private_prompt_character}"
        )
        await choose_prompt.finish(msg)

    async def handle_group_prompt(arg_list: list[str]) -> None:
        """处理群组提示词预设"""
        if len(arg_list) >= 2:
            for i in config_manager.get_prompts().group:
                if i.name == arg_list[1]:
                    config_manager.config.group_prompt_character = i.name
                    config_manager.load_prompt()
                    config_manager.save_config()
                    await choose_prompt.finish(f"已设置群组的提示词预设为：{i.name}")
            await choose_prompt.finish(
                "未找到预设，请输入/choose_prompt group查看预设列表"
            )
        else:
            await list_available_prompts(config_manager.get_prompts().group, "group")

    async def handle_private_prompt(arg_list: list[str]) -> None:
        """处理私聊提示词预设"""
        if len(arg_list) >= 2:
            for i in config_manager.get_prompts().private:
                if i.name == arg_list[1]:
                    config_manager.config.private_prompt_character = i.name
                    config_manager.load_prompt()
                    config_manager.save_config()
                    await choose_prompt.finish(f"已设置私聊的提示词预设为：{i.name}")
            await choose_prompt.finish(
                "未找到预设，请输入/choose_prompt private查看预设列表"
            )
        else:
            await list_available_prompts(
                config_manager.get_prompts().private, "private"
            )

    async def list_available_prompts(prompts: list[Any], prompt_type: str) -> None:
        """列出可用的提示词预设"""
        msg = "可选的预设名称：\n"
        for index, i in enumerate(prompts):
            current_marker = (
                " (当前)"
                if (
                    prompt_type == "group"
                    and i.name == config_manager.config.group_prompt_character
                )
                or (
                    prompt_type == "private"
                    and i.name == config_manager.config.private_prompt_character
                )
                else ""
            )
            msg += f"{index + 1}). {i.name}{current_marker}\n"
        await choose_prompt.finish(msg)

    if not config_manager.config.enable:
        choose_prompt.skip()

    if event.user_id not in config_manager.config.admins:
        await choose_prompt.finish("只有管理员才能设置预设。")

    arg_list = args.extract_plain_text().strip().split()

    if not arg_list:
        await display_current_prompts()
        return

    if arg_list[0] == "group":
        await handle_group_prompt(arg_list)
    elif arg_list[0] == "private":
        await handle_private_prompt(arg_list)


@sessions.handle()
async def sessions_handle(bot: Bot, event: MessageEvent, args: Message = CommandArg()):
    """处理会话管理命令的入口函数"""

    async def display_sessions(data: dict) -> None:
        """显示历史会话列表"""
        if not data.get("sessions"):
            await sessions.finish("没有历史会话")
        message_content = "历史会话\n"
        for index, msg in enumerate(data["sessions"]):
            message_content += f"编号：{index}) ：{msg['messages'][0]['content'][9:]}... 时间：{datetime.fromtimestamp(msg['time']).strftime('%Y-%m-%d %I:%M:%S %p')}\n"
        await sessions.finish(message_content)

    async def set_session(data: dict, arg_list: list[str], event: MessageEvent) -> None:
        """覆盖当前会话为指定编号的会话"""
        try:
            if len(arg_list) >= 2:
                data["memory"]["messages"] = data["sessions"][int(arg_list[1])][
                    "messages"
                ]
                data["timestamp"] = time.time()
                write_memory_data(event, data)
                await sessions.send("完成记忆覆盖。")
            else:
                await sessions.finish("请输入正确编号")
        except Exception:
            await sessions.finish("覆盖记忆文件失败，这个对话可能损坏了。")

    async def delete_session(
        data: dict, arg_list: list[str], event: MessageEvent
    ) -> None:
        """删除指定编号的会话"""
        try:
            if len(arg_list) >= 2:
                data["sessions"].remove(data["sessions"][int(arg_list[1])])
                write_memory_data(event, data)
            else:
                await sessions.finish("请输入正确编号")
        except Exception:
            await sessions.finish("删除指定编号会话失败。")

    async def archive_session(data: dict, event: MessageEvent) -> None:
        """归档当前会话"""
        try:
            if data["memory"]["messages"]:
                data["sessions"].append(
                    {"messages": data["memory"]["messages"], "time": time.time()}
                )
                data["memory"]["messages"] = []
                data["timestamp"] = time.time()
                write_memory_data(event, data)
                await sessions.finish("当前会话已归档。")
            else:
                await sessions.finish("当前对话为空！")
        except Exception:
            await sessions.finish("归档当前会话失败。")

    async def clear_sessions(data: dict, event: MessageEvent) -> None:
        """清空所有会话"""
        try:
            data["sessions"] = []
            data["timestamp"] = time.time()
            write_memory_data(event, data)
            await sessions.finish("会话已清空。")
        except Exception:
            await sessions.finish("清空当前会话失败。")

    if not config_manager.config.session_control:
        sessions.skip()

    data = get_memory_data(event)

    if isinstance(event, GroupMessageEvent) and (
        (
            await bot.get_group_member_info(
                group_id=event.group_id, user_id=event.user_id
            )
        )["role"]
        == "member"
        and event.user_id not in config_manager.config.admins
    ):
        await sessions.finish("你没有操作历史会话的权限")

    arg_list = args.extract_plain_text().strip().split()

    if not arg_list:
        await display_sessions(data)

    command = arg_list[0]
    if command == "set":
        await set_session(data, arg_list, event)
    elif command == "del":
        await delete_session(data, arg_list, event)
    elif command == "archive":
        await archive_session(data, event)
    elif command == "clear":
        await clear_sessions(data, event)
    elif command == "help":
        await sessions.finish(
            "Sessions指令帮助：\nset：覆盖当前会话为指定编号的会话\ndel：删除指定编号的会话\narchive：归档当前会话\nclear：清空所有会话\n"
        )
    else:
        await sessions.finish("未知命令，请输入/help查看帮助。")


@set_preset.handle()
async def _(event: MessageEvent, args: Message = CommandArg()):
    # 检查插件是否启用
    if not config_manager.config.enable:
        set_preset.skip()

    # 检查用户是否为管理员
    if event.user_id not in config_manager.config.admins:
        await set_preset.finish("只有管理员才能设置预设。")

    # 提取命令参数
    arg = args.extract_plain_text().strip()

    # 如果参数不为空
    if arg != "":
        # 遍历模型列表
        for i in config_manager.get_models():
            # 如果模型名称与参数匹配
            if i.name == arg:
                # 设置预设并保存配置
                config_manager.config.preset = i.name
                config_manager.save_config()
                # 回复设置成功
                await set_preset.finish(f"已设置预设为：{i.name}，模型：{i.model}")
                break
        else:
            # 如果未找到预设，提示用户
            await set_preset.finish("未找到预设，请输入/presets查看预设列表。")
    else:
        # 如果参数为空，重置预设为默认
        config_manager.config.preset = "__main__"
        config_manager.save_config()
        # 回复重置成功
        await set_preset.finish(
            f"已重置预设为：主配置文件，模型：{config_manager.config.model}"
        )


@presets.handle()
async def _(event: MessageEvent):
    """处理模型预设查看的事件处理器"""
    # 检查功能是否已启用，未启用则跳过处理
    if not config_manager.config.enable:
        presets.skip()

    # 检查用户是否为管理员，非管理员则发送消息并结束处理
    if event.user_id not in config_manager.config.admins:
        await presets.finish("只有管理员才能查看模型预设。")

    # 构建消息字符串，包含当前模型预设信息
    msg = f"模型预设:\n当前：{'主配置文件' if config_manager.config.preset == '__main__' else config_manager.config.preset}\n主配置文件：{config_manager.config.model}"

    # 遍历模型列表，添加每个预设的名称和模型到消息字符串
    for i in config_manager.get_models():
        msg += f"\n预设名称：{i.name}，模型：{i.model}"

    # 发送消息给用户并结束处理
    await presets.finish(msg)


@prompt.handle()
async def _(bot: Bot, event: GroupMessageEvent, args: Message = CommandArg()):
    """处理prompt命令的异步函数。此函数根据不同的条件和用户输入来管理prompt的设置和查询"""
    # 检查是否启用prompt功能，未启用则跳过处理
    if not config_manager.config.enable:
        prompt.skip()
    # 检查是否允许自定义prompt，不允许则结束处理
    if not config_manager.config.allow_custom_prompt:
        await prompt.finish("当前不允许自定义prompt。")

    # 检查用户是否为群成员且非管理员，是则结束处理
    if (
        await is_member(event, bot)
        and event.user_id not in config_manager.config.admins
    ):
        await prompt.finish("群成员不能设置prompt.")

    data = get_memory_data(event)
    arg = args.extract_plain_text().strip()

    # 检查输入长度是否过长，过长则提示用户并返回
    if len(arg) >= 1000:
        await prompt.send("prompt过长，预期的参数不超过1000字。")
        return

    # 检查输入是否为空，为空则提示用户如何使用命令
    if arg.strip() == "":
        await prompt.send(
            "请输入prompt或参数（--(show) 展示当前提示词，--(clear) 清空当前prompt，--(set) [文字]则设置提示词，e.g.:/prompt --(show)）,/prompt --(set) [text]。）"
        )
        return

    # 根据用户输入的不同命令进行相应的处理
    if arg.startswith("--(show)"):
        await prompt.send(f"Prompt:\n{data.get('prompt', '未设置prompt')}")
        return
    elif arg.startswith("--(clear)"):
        data["prompt"] = ""
        await prompt.send("prompt已清空。")
    elif arg.startswith("--(set)"):
        arg = arg.replace("--(set)", "").strip()
        data["prompt"] = arg
        await prompt.send(f"prompt已设置为：\n{arg}")
    else:
        await prompt.send(
            "请输入prompt或参数（--(show) 展示当前提示词，--(clear) 清空当前prompt，--(set) [文字]则设置提示词，e.g.:/prompt --(show)）,/prompt --(set) [text]。"
        )
        return

    # 更新记忆数据
    write_memory_data(event, data)


# 当有人加入群聊时触发的事件处理函数
@add_notice.handle()
async def _(event: GroupIncreaseNoticeEvent):
    """处理群聊增加通知事件的异步函数"""
    if not config_manager.config.enable:
        add_notice.skip()
    # 检查配置，如果不发送被邀请后的消息，则直接返回
    if not config_manager.config.send_msg_after_be_invited:
        return
    # 如果事件的用户ID与机器人自身ID相同，表示机器人被邀请加入群聊
    if event.user_id == event.self_id:
        # 发送配置中的群聊添加消息
        await add_notice.send(config_manager.config.group_added_msg)
        return


# 处理调试模式开关的函数
@debug_switch.handle()
async def _(event: MessageEvent, matcher: Matcher):
    """根据用户权限开启或关闭调试模式"""
    if not config_manager.config.enable:
        matcher.skip()
    # 如果不是管理员用户，直接返回
    if event.user_id not in config_manager.config.admins:
        return
    # 根据当前调试模式状态，开启或关闭调试模式，并发送通知
    if chat_manager.debug:
        chat_manager.debug = False
        await debug_switch.finish(
            "已关闭调试模式（该模式适用于开发者，如果你作为普通用户使用，请关闭调试模式）"
        )
    else:
        chat_manager.debug = True
        await debug_switch.finish(
            "已开启调试模式（该模式适用于开发者，如果你作为普通用户使用，请关闭调试模式）"
        )


# 当有消息撤回时触发处理函数
@recall.handle()
async def _(bot: Bot, event: GroupRecallNoticeEvent, matcher: Matcher):
    """处理消息撤回通知事件"""

    # 检查是否启用了插件功能，未启用则跳过后续处理
    if not config_manager.config.enable:
        matcher.skip()
    # 通过随机数决定是否响应，增加趣味性和减少响应频率
    if random.randint(1, 3) != 2:
        return
    # 检查配置中是否允许在删除自己的消息后发言，不允许则直接返回
    if not config_manager.config.say_after_self_msg_be_deleted:
        return
    # 判断事件是否为机器人自己删除了自己的消息
    if event.user_id == event.self_id:
        # 如果是机器人自己删除了自己的消息，并且操作者也是机器人自己，则不进行回复
        if event.operator_id == event.self_id:
            return
        # 从配置中获取删除消息后可能的回复内容
        recallmsg = config_manager.config.after_deleted_say_what
        # 从预设的回复内容中随机选择一条发送
        await recall.send(random.choice(recallmsg))
        return


# 处理菜单命令的函数
@menu.handle()
async def _(matcher: Matcher):
    """处理聊天菜单命令"""
    if not config_manager.config.enable:
        matcher.skip()

    # 初始化消息内容为默认菜单消息
    msg = chat_manager.menu_msg

    # 遍历自定义菜单项，添加到消息内容中
    for menus in chat_manager.custom_menu:
        msg += f"\n{menus['cmd']} {menus['describe']}"

    # 根据配置信息，添加群聊或私聊聊天可用性的提示信息
    msg += f"\n{'群内可以at我与我聊天，' if config_manager.config.enable_group_chat else '未启用群内聊天，'}{'在私聊可以直接聊天。' if config_manager.config.enable_group_chat else '未启用私聊聊天'}\nPowered by Suggar chat plugin"

    # 发送最终的消息内容
    await menu.send(msg)


@poke.handle()
async def _(event: PokeNotifyEvent, bot: Bot, matcher: Matcher):
    """处理戳一戳事件"""

    async def handle_group_poke(event: PokeNotifyEvent, bot: Bot):
        """处理群聊中的戳一戳事件"""
        Group_Data = get_memory_data(event)
        if not Group_Data["enable"]:
            return

        user_name = (
            await bot.get_group_member_info(
                group_id=event.group_id, user_id=event.user_id
            )
        )["nickname"]

        send_messages = [
            {"role": "system", "content": f"{config_manager.group_train}"},
            {
                "role": "user",
                "content": f"\\（戳一戳消息\\){user_name} (QQ:{event.user_id}) 戳了戳你",
            },
        ]

        response = await process_poke_event(event, send_messages)
        message = (
            MessageSegment.at(user_id=event.user_id)
            + MessageSegment.text(" ")
            + MessageSegment.text(response)
        )

        if not config_manager.config.nature_chat_style:
            await poke.send(message)
        else:
            await send_split_messages(response, event.user_id)

    async def handle_private_poke(event: PokeNotifyEvent, bot: Bot):
        """处理私聊中的戳一戳事件"""
        name = get_friend_info(event.user_id, bot)
        send_messages = [
            {"role": "system", "content": f"{config_manager.private_train}"},
            {
                "role": "user",
                "content": f" \\（戳一戳消息\\) {name}(QQ:{event.user_id}) 戳了戳你",
            },
        ]

        response = await process_poke_event(event, send_messages)
        if not config_manager.config.nature_chat_style:
            await poke.send(MessageSegment.text(response))
        else:
            await send_split_messages(response, event.user_id)

    async def process_poke_event(event: PokeNotifyEvent, send_messages: list) -> str:
        """处理戳一戳事件的核心逻辑"""
        if config_manager.config.matcher_function:
            _matcher = SuggarMatcher(event_type=EventType().before_poke())
            poke_event = PokeEvent(
                nbevent=event,
                send_message=send_messages,
                model_response=[""],
                user_id=event.user_id,
            )
            await _matcher.trigger_event(poke_event, _matcher)
            send_messages = poke_event.get_send_message()

        response = await get_chat(send_messages)

        if config_manager.config.matcher_function:
            _matcher = SuggarMatcher(event_type=EventType().poke())
            poke_event = PokeEvent(
                nbevent=event,
                send_message=send_messages,
                model_response=[response],
                user_id=event.user_id,
            )
            await _matcher.trigger_event(poke_event, _matcher)
            response = poke_event.model_response

        if chat_manager.debug:
            await send_to_admin(f"POKEMSG {send_messages}")

        return response

    async def send_split_messages(response: str, user_id: int):
        """发送分段消息"""
        if response_list := split_message_into_chats(response):
            first_message = (
                MessageSegment.at(user_id) + MessageSegment.text(" ") + response_list[0]
            )
            await poke.send(first_message)

            for message in response_list[1:]:
                await poke.send(message)
                await asyncio.sleep(
                    random.randint(1, 3) + len(message) // random.randint(80, 100)
                )

    async def handle_poke_exception():
        """处理戳一戳事件中的异常"""
        exc_type, exc_value, exc_traceback = sys.exc_info()
        logger.error(
            f"Exception type: {exc_type.__name__}"
            if exc_type
            else "Exception type: None"
        )
        logger.error(f"Exception message: {exc_value!s}")
        import traceback

        await send_to_admin(f"出错了！{exc_value},\n{exc_type!s}")
        await send_to_admin(f"{traceback.format_exc()}")

        logger.error(
            f"Detailed exception info:\n{''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))}"
        )

    if not config_manager.config.enable or not config_manager.config.poke_reply:
        matcher.skip()
        return

    if event.target_id != event.self_id:
        return

    try:
        if event.group_id is not None:
            await handle_group_poke(event, bot)
        else:
            await handle_private_poke(event, bot)
    except Exception:
        await handle_poke_exception()


@disable.handle()
async def _(bot: Bot, event: GroupMessageEvent, matcher: Matcher):
    """处理禁用聊天功能的异步函数"""
    # 检查全局配置是否启用，如果未启用则跳过后续处理
    if not config_manager.config.enable:
        matcher.skip()

    # 获取发送消息的成员信息
    member = await bot.get_group_member_info(
        group_id=event.group_id, user_id=event.user_id
    )

    # 检查成员是否为普通成员且不在管理员列表中，如果是则发送提示消息并返回
    if member["role"] == "member" and event.user_id not in config_manager.config.admins:
        await disable.send("你没有这样的力量呢～（管理员/管理员+）")
        return

    # 记录禁用操作的日志
    logger.debug(f"{event.group_id} disabled")

    # 获取并更新记忆中的数据结构
    data = get_memory_data(event)
    if data["id"] == event.group_id:
        if not data["enable"]:
            await disable.send("聊天禁用")
        else:
            data["enable"] = False
            await disable.send("聊天已经禁用")

    # 将更新后的数据结构写回记忆
    write_memory_data(event, data)


@enable.handle()
async def _(bot: Bot, event: GroupMessageEvent, matcher: Matcher):
    """处理启用聊天功能的命令"""
    # 检查全局配置，如果未启用则跳过后续处理
    if not config_manager.config.enable:
        matcher.skip()

    # 获取发送命令的用户在群中的角色信息
    member = await bot.get_group_member_info(
        group_id=event.group_id, user_id=event.user_id
    )
    # 如果用户是普通成员且不在管理员列表中，则发送提示信息并返回
    if member["role"] == "member" and event.user_id not in config_manager.config.admins:
        await enable.send("你没有这样的力量呢～（管理员/管理员+）")
        return

    # 记录日志
    logger.debug(f"{event.group_id}enabled")
    # 获取记忆中的数据
    data = get_memory_data(event)
    # 检查记忆数据是否与当前群组相关
    if data["id"] == event.group_id:
        # 如果聊天功能已启用，则发送提示信息
        if not data["enable"]:
            # 如果聊天功能未启用，则启用并发送提示信息
            data["enable"] = True
        await enable.send("聊天启用")
    # 更新记忆数据
    write_memory_data(event, data)


@del_memory.handle()
async def _(bot: Bot, event: MessageEvent, matcher: Matcher):
    """处理删除记忆指令"""
    # 检查配置以确定是否启用功能
    if not config_manager.config.enable:
        matcher.skip()

    # 判断事件是否来自群聊
    if isinstance(event, GroupMessageEvent):
        # 获取群成员信息
        member = await bot.get_group_member_info(
            group_id=event.group_id, user_id=event.user_id
        )

        # 检查用户权限，非管理员且不在管理员列表中的用户将被拒绝
        if (
            member["role"] == "member"
            and event.user_id not in config_manager.config.admins
        ):
            await del_memory.send("你没有这样的力量（管理员/管理员+）")
            return

        # 获取群聊记忆数据
        GData = get_memory_data(event)

        # 清除群聊上下文
        if GData["id"] == event.group_id:
            GData["memory"]["messages"] = []
            await del_memory.send("上下文已清除")
            write_memory_data(event, GData)
            logger.debug(f"{event.group_id}Memory deleted")

    else:
        # 获取私聊记忆数据
        FData = get_memory_data(event)

        # 清除私聊上下文
        if FData["id"] == event.user_id:
            FData["memory"]["messages"] = []
            await del_memory.send("上下文已清除")
            logger.debug(f"{event.user_id}Memory deleted")
            write_memory_data(event, FData)


@chat.handle()
async def _(event: MessageEvent, matcher: Matcher, bot: Bot):
    """处理聊天事件"""

    async def handle_group_message(
        event: GroupMessageEvent,
        matcher: Matcher,
        bot: Bot,
        group_data: dict,
        memory_length_limit: int,
        Date: str,
    ):
        if not config_manager.config.enable_group_chat:
            matcher.skip()

        if not group_data["enable"]:
            await chat.send("聊天没有启用")
            return

        await manage_sessions(event, group_data, chat_manager.session_clear_group)

        group_id = event.group_id
        user_id = event.user_id
        user_name = (
            await bot.get_group_member_info(group_id=group_id, user_id=user_id)
        )["nickname"]
        content = await synthesize_message(event.get_message(), bot)

        if content.strip() == "":
            content = ""

        role = await get_user_role(bot, group_id, user_id)
        logger.debug(f"{Date}{user_name}（{user_id}）说:{content}")

        if event.reply:
            content = await handle_reply(event.reply, bot, group_id, content)

        group_data["memory"]["messages"].append(
            {
                "role": "user",
                "content": f"[{role}][{Date}][{user_name}（{user_id}）]说:{content if config_manager.config.parse_segments else event.message.extract_plain_text()}",
            }
        )

        await enforce_memory_limit(group_data, memory_length_limit)
        await enforce_token_limit(group_data, config_manager.group_train)

        send_messages = prepare_send_messages(group_data, config_manager.group_train)
        response = await process_chat(event, send_messages)

        group_data["memory"]["messages"].append(
            {"role": "assistant", "content": str(response)}
        )
        await send_response(event, response)

        write_memory_data(event, group_data)

    async def handle_private_message(
        event: PrivateMessageEvent,
        matcher: Matcher,
        bot: Bot,
        private_data: dict,
        memory_length_limit: int,
        Date: str,
    ):
        if not config_manager.config.enable_private_chat:
            matcher.skip()

        await manage_sessions(event, private_data, chat_manager.session_clear_user)

        content = await synthesize_message(event.get_message(), bot)

        if content.strip() == "":
            content = ""

        if event.reply:
            content = await handle_reply(event.reply, bot, None, content)

        private_data["memory"]["messages"].append(
            {
                "role": "user",
                "content": f"{Date}{await get_friend_info(event.user_id, bot=bot)}（{event.user_id}）： {str(content) if config_manager.config.parse_segments else event.message.extract_plain_text()}",
            }
        )

        await enforce_memory_limit(private_data, memory_length_limit)
        await enforce_token_limit(private_data, config_manager.private_train)

        send_messages = prepare_send_messages(
            private_data, config_manager.private_train
        )
        response = await process_chat(event, send_messages)

        private_data["memory"]["messages"].append(
            {"role": "assistant", "content": str(response)}
        )
        await send_response(event, response)

        write_memory_data(event, private_data)

    async def manage_sessions(
        event: GroupMessageEvent | PrivateMessageEvent,
        data: dict,
        session_clear_list: list,
    ):
        if data.get("sessions") is None:
            data["sessions"] = []
        if data.get("timestamp") is None:
            data["timestamp"] = time.time()

        if config_manager.config.session_control:
            for session in session_clear_list:
                if session["id"] == (
                    event.group_id
                    if isinstance(event, GroupMessageEvent)
                    else event.user_id
                ):
                    if not event.reply:
                        session_clear_list.remove(session)
                    break

            if (time.time() - data["timestamp"]) >= (
                config_manager.config.session_control_time * 60
            ):
                data["sessions"].append(
                    {"messages": data["memory"]["messages"], "time": time.time()}
                )
                while (
                    len(data["sessions"])
                    > config_manager.config.session_control_history
                ):
                    data["sessions"].remove(data["sessions"][0])
                data["memory"]["messages"] = []
                data["timestamp"] = time.time()
                write_memory_data(event, data)
                chated = await chat.send(
                    f'如果想和我继续用刚刚的上下文聊天，快回复我✨"继续"✨吧！\n（超过{config_manager.config.session_control_time}分钟没理我我就会被系统抱走存档哦！）'
                )
                session_clear_list.append(
                    {
                        "id": (
                            event.group_id
                            if isinstance(event, GroupMessageEvent)
                            else event.user_id
                        ),
                        "message_id": chated["message_id"],
                        "timestamp": time.time(),
                    }
                )
                return
            elif event.reply:
                for session in session_clear_list:
                    if (
                        session["id"]
                        == (
                            event.group_id
                            if isinstance(event, GroupMessageEvent)
                            else event.user_id
                        )
                        and "继续" in event.reply.message.extract_plain_text()
                    ):
                        with contextlib.suppress(Exception):
                            if time.time() - session["timestamp"] < 100:
                                await bot.delete_msg(message_id=session["message_id"])
                        session_clear_list.remove(session)
                        data["memory"]["messages"] = data["sessions"][-1]["messages"]
                        data["sessions"].pop()
                        await chat.send("让我们继续聊天吧～")
                        return write_memory_data(event, data)

    async def handle_reply(
        reply: Reply, bot: Bot, group_id: int | None, content: str
    ) -> str:
        if not reply.sender.user_id:
            return content
        dt_object = datetime.fromtimestamp(reply.time)
        weekday = dt_object.strftime("%A")
        formatted_time = dt_object.strftime("%Y-%m-%d %I:%M:%S %p")
        role = (
            await get_user_role(bot, group_id, reply.sender.user_id) if group_id else ""
        )
        reply_content = await synthesize_message(reply.message, bot)
        return f"{content}\n（（（引用的消息）））：\n{formatted_time} {weekday} [{role}]{reply.sender.nickname}（QQ:{reply.sender.user_id}）说：{reply_content}"

    async def get_user_role(bot: Bot, group_id: int, user_id: int) -> str:
        role = (await bot.get_group_member_info(group_id=group_id, user_id=user_id))[
            "role"
        ]
        return {"admin": "群管理员", "owner": "群主", "member": "普通成员"}.get(
            role, "[获取身份失败]"
        )

    async def enforce_memory_limit(data: dict, memory_length_limit: int):
        while len(data["memory"]["messages"]) > memory_length_limit or (
            data["memory"]["messages"][0]["role"] != "user"
        ):
            del data["memory"]["messages"][0]

    async def enforce_token_limit(data: dict, train: dict):
        if config_manager.config.enable_tokens_limit:
            memory_l = [train.copy(), *data["memory"]["messages"].copy()]
            full_string = "".join(st["content"] for st in memory_l)
            tokens = hybrid_token_count(
                full_string, config_manager.config.tokens_count_mode
            )
            while tokens > config_manager.config.session_max_tokens:
                del data["memory"]["messages"][0]
                full_string = "".join(
                    st["content"]
                    for st in [train.copy(), *data["memory"]["messages"].copy()]
                )
                tokens = hybrid_token_count(
                    full_string, config_manager.config.tokens_count_mode
                )

    def prepare_send_messages(data: dict, train: dict) -> list:
        train["content"] += (
            f"\n以下是一些补充内容，如果与上面任何一条有冲突请忽略。\n{data.get('prompt', '无')}"
        )
        send_messages = data["memory"]["messages"].copy()
        send_messages.insert(0, train)
        return send_messages

    async def process_chat(event: MessageEvent, send_messages: list) -> str:
        if config_manager.config.matcher_function:
            _matcher = SuggarMatcher(event_type=EventType().before_chat())
            chat_event = ChatEvent(
                nbevent=event,
                send_message=send_messages,
                model_response=[""],
                user_id=event.user_id,
            )
            await _matcher.trigger_event(chat_event, _matcher)
            send_messages = chat_event.get_send_message()

        response = await get_chat(send_messages)

        if config_manager.config.matcher_function:
            _matcher = SuggarMatcher(event_type=EventType().chat())
            chat_event = ChatEvent(
                nbevent=event,
                send_message=send_messages,
                model_response=[response],
                user_id=event.user_id,
            )
            await _matcher.trigger_event(chat_event, _matcher)
            response = chat_event.model_response

        return response

    async def send_response(event: MessageEvent, response: str):
        if not config_manager.config.nature_chat_style:
            await chat.send(
                MessageSegment.reply(event.message_id) + MessageSegment.text(response)
            )
        elif response_list := split_message_into_chats(response):
            first_message = (
                MessageSegment.at(event.user_id)
                + MessageSegment.text(" ")
                + response_list[0]
            )
            await chat.send(first_message)
            for message in response_list[1:]:
                await chat.send(message)
                await asyncio.sleep(
                    random.randint(1, 3) + len(message) // random.randint(80, 100)
                )

    async def handle_exception():
        await chat.send("出错了稍后试试吧（错误已反馈）")
        exc_type, exc_value, exc_traceback = sys.exc_info()
        logger.error(
            f"Exception type: {exc_type.__name__}"
            if exc_type
            else "Exception type: None"
        )
        logger.error(f"Exception message: {exc_value!s}")
        import traceback

        await send_to_admin(f"出错了！{exc_value},\n{exc_type!s}")
        await send_to_admin(f"{traceback.format_exc()}")
        logger.error(
            f"Detailed exception info:\n{''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))}"
        )

    if not config_manager.config.enable:
        matcher.skip()

    memory_length_limit = config_manager.config.memory_lenth_limit
    Date = get_current_datetime_timestamp()
    bot = nonebot.get_bot()

    if event.message.extract_plain_text().strip().startswith("/"):
        matcher.skip()

    if event.message.extract_plain_text().startswith("菜单"):
        await matcher.finish(chat_manager.menu_msg)

    group_data = get_memory_data(event)
    private_data = get_memory_data(event)

    try:
        if isinstance(event, GroupMessageEvent):
            await handle_group_message(
                event, matcher, bot, group_data, memory_length_limit, Date
            )
        elif isinstance(event, PrivateMessageEvent):
            await handle_private_message(
                event, matcher, bot, private_data, memory_length_limit, Date
            )
        else:
            matcher.skip()
    except NoneBotException as e:
        raise e
    except Exception:
        await handle_exception()
