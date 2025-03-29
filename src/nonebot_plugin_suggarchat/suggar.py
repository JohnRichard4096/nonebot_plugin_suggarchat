import asyncio
import random
import sys
import time
from collections.abc import Callable, Coroutine
from datetime import datetime
from typing import Any

import nonebot.adapters
import openai
from nonebot import logger, on_command, on_message, on_notice
from nonebot.adapters import Bot, Message
from nonebot.adapters.onebot.v11 import (
    GroupIncreaseNoticeEvent,
    GroupMessageEvent,
    GroupRecallNoticeEvent,
    MessageEvent,
    MessageSegment,
    PokeNotifyEvent,
    PrivateMessageEvent,
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
    format_datetime_timestamp,
    get_current_datetime_timestamp,
    get_friend_info,
    get_memory_data,
    hybrid_token_count,
    split_message_into_chats,
    synthesize_message,
    write_memory_data,
)

debug = False

session_clear_group = []
session_clear_user = []
custom_menu = []

running_messages = {}
running_messages_poke = {}


async def openai_get_chat(
    base_url, model, key, messages, max_tokens, config: Config, bot: Bot
) -> str:
    # 记录日志，开始获取对话
    logger.debug(f"Start to get response with model {model}")
    logger.debug(f"Preset：{config.preset}")
    logger.debug(f"Key：{key[:7]}...")
    logger.debug(f"API base_url：{base_url}")
    if (
        not str(config.open_ai_base_url).strip()
        or not str(config.open_ai_api_key).strip()
    ):
        raise RuntimeError("错误！OpenAI Url或Key为空！")
    client = openai.AsyncOpenAI(
        base_url=base_url, api_key=key, timeout=config.llm_timeout
    )
    # 创建聊天完成请求
    for i in range(3):
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
            await send_to_admin(f"在获取对话时发生了错误: {e}", bot)
            logger.info(f"尝试第{i + 1}次重试")
            continue
    response = ""
    if config.stream and isinstance(completion, openai.AsyncStream):
        # 流式接收响应并构建最终的聊天文本
        async for chunk in completion:
            try:
                if chunk.choices[0].delta.content is not None:
                    response += chunk.choices[0].delta.content
                    if debug:
                        logger.debug(chunk.choices[0].delta.content)
            except IndexError:
                break
        # 记录生成的响应日志
    else:
        if debug:
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


async def get_chat(messages: list, bot: Bot | None = None) -> str:
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

    # 根据配置中的预设值，选择不同的API密钥和基础URL
    if config_manager.config.preset == "__main__":
        # 如果是主配置，直接使用配置文件中的设置
        base_url = config_manager.config.open_ai_base_url
        key = config_manager.config.open_ai_api_key
        model = config_manager.config.model
    else:
        # 如果是其他预设，从模型列表中查找匹配的设置
        for i in config_manager.models:
            if i.name == config_manager.config.preset:
                base_url = i.base_url
                key = i.api_key
                model = i.model
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
            # 保存更新后的配置
            config_manager.save_config()
    if config_manager.config.protocol == "__main__":
        return await openai_get_chat(
            base_url,
            model,
            key,
            messages,
            max_tokens,
            config_manager.config,
            bot or nonebot.get_bot(),
        )
    elif config_manager.config.protocol not in protocols_adapters:
        raise Exception(f"协议 {config_manager.config.protocol} 的适配器未找到!")
    else:
        return await protocols_adapters[config_manager.config.protocol](
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


@sessions.handle()
async def sessions_handle(bot: Bot, event: MessageEvent, args: Message = CommandArg()):
    """处理会话管理命令的入口函数

    Args:
        bot: 机器人实例，用于执行API调用
        event: 消息事件对象，包含事件上下文信息
        args: 用户输入的命令参数，通过Message类型接收

    功能说明：
        实现历史会话的查看、覆盖、删除、归档等管理操作
        支持群组管理员和配置中的管理员用户操作
    """
    # 检查全局会话控制开关
    if not config_manager.config.session_control:
        sessions.skip()

    # 获取当前用户/群组的记忆数据
    data = get_memory_data(event)

    # 权限验证流程（仅群组消息需要验证）
    if isinstance(event, GroupMessageEvent) and ((
                await bot.get_group_member_info(
                    group_id=event.group_id, user_id=event.user_id
                )
            )["role"] == "member" and event.user_id not in config_manager.config.admins):
        await sessions.finish("你没有操作历史会话的权限")
    # 解析输入参数
    arg_list = args.extract_plain_text().strip().split()

    # 无参数时显示会话列表
    if args.extract_plain_text().strip() == "":
        message_content = "历史会话\n"
        if data.get("sessions") is None:
            await sessions.finish("没有历史会话")
        # 构建会话列表消息
        for msg in data["sessions"]:
            message_content += f"编号：{data['sessions'].index(msg)} ：{msg['messages'][0][9:]}... 时间：{format_datetime_timestamp(msg['time'])}\n"
        await sessions.finish(message_content)

    # 处理带参数的命令
    if len(arg_list) >= 1:
        # 覆盖当前会话命令
        if arg_list[0] == "set":
            try:
                if len(arg_list) >= 2:
                    data["memory"]["messages"] = data["sessions"][int(arg_list[1])]
                    data["timestamp"] = time.time()
                    write_memory_data(event, data)
                    await sessions.send("完成记忆覆盖。")
                else:
                    await sessions.finish("请输入正确编号")
            except NoneBotException as e:
                raise e
            except Exception:
                await sessions.finish("覆盖记忆文件失败。")

        # 删除会话命令
        elif arg_list[0] == "del":
            try:
                if len(arg_list) >= 2:
                    data["sessions"].remove(data["sessions"][int(arg_list[1])])
                    write_memory_data(event, data)
                else:
                    await sessions.finish("请输入正确编号")
            except NoneBotException as e:
                raise e
            except Exception:
                await sessions.finish("删除指定编号会话失败。")

        # 归档当前会话命令
        elif arg_list[0] == "archive":
            try:
                data["sessions"].append(data["memory"]["messages"])
                data["memory"]["messages"] = []
                data["timestamp"] = time.time()
                write_memory_data(event, data)
                await sessions.finish("当前会话已归档。")
            except NoneBotException as e:
                raise e
            except Exception:
                await sessions.finish("归档当前会话失败。")

        # 清空会话命令
        elif arg_list[0] == "clear":
            try:
                data["sessions"] = []
                data["timestamp"] = time.time()
                write_memory_data(event, data)
                await sessions.finish("会话已清空。")
            except NoneBotException as e:
                raise e
            except Exception:
                await sessions.finish("清空当前会话失败。")

        # 帮助命令
        elif arg_list[0] == "help":
            await sessions.finish(
                "Sessions指令帮助：\nset：覆盖当前会话为指定编号的会话\ndel：删除指定编号的会话\narchive：归档当前会话\nclear：清空所有会话\n"
            )


"""
@del_all_memory.handle()
async def del_all_memory_handle(bot:Bot,event:MessageEvent):
    global config
    if not event.user_id in config['admins']:
        await del_all_memory.finish("你没有权限执行此操作")
"""


@set_preset.handle()
async def _(bot: Bot, event: MessageEvent, args: Message = CommandArg()):
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
        for i in config_manager.models:
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
        await set_preset.finish(f"已重置预设为：主配置文件，模型：{config_manager.config.model}")


@presets.handle()
async def _(bot: Bot, event: MessageEvent):
    """
    处理预设命令的异步函数。

    该函数响应预设命令，检查用户是否为管理员，然后返回当前模型预设的信息。

    参数:
    - bot: Bot对象，用于与平台交互。
    - event: MessageEvent对象，包含事件相关的信息。

    使用的全局变量:
    - admins: 包含管理员用户ID的列表。
    - config: 包含配置信息的字典。
    - ifenable: 布尔值，指示功能是否已启用。

    """

    # 检查功能是否已启用，未启用则跳过处理
    if not config_manager.config.enable:
        presets.skip()

    # 检查用户是否为管理员，非管理员则发送消息并结束处理
    if event.user_id not in config_manager.config.admins:
        await presets.finish("只有管理员才能查看模型预设。")

    # 构建消息字符串，包含当前模型预设信息
    msg = f"模型预设:\n当前：{'主配置文件' if config_manager.config.preset == '__main__' else config_manager.config.preset}\n主配置文件：{config_manager.config.model}"

    # 遍历模型列表，添加每个预设的名称和模型到消息字符串
    for i in config_manager.models:
        msg += f"\n预设名称：{i.name}，模型：{i.model}"

    # 发送消息给用户并结束处理
    await presets.finish(msg)


@prompt.handle()
async def _(bot: Bot, event: GroupMessageEvent, args: Message = CommandArg()):
    """
    处理prompt命令的异步函数。此函数根据不同的条件和用户输入来管理prompt的设置和查询。

    参数:
    - bot: Bot对象，用于发送消息和与机器人交互。
    - event: GroupMessageEvent对象，包含事件的详细信息，如用户ID和消息内容。
    - args: Message对象，包含用户输入的命令参数。

    返回值:
    无返回值。
    """
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
        return

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
async def _(bot: Bot, event: GroupIncreaseNoticeEvent):
    """
    处理群聊增加通知事件的异步函数。

    参数:
    - bot: Bot对象，用于访问和操作机器人。
    - event: GroupIncreaseNoticeEvent对象，包含事件相关信息。

    此函数主要用于处理当机器人所在的群聊中增加新成员时的通知事件。
    它会根据全局配置变量config中的设置决定是否发送欢迎消息。
    """
    # 检查全局配置，如果未启用，则跳过处理
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
async def _(bot: Bot, event: MessageEvent, matcher: Matcher):
    """
    根据用户权限开启或关闭调试模式。

    参数:
    - bot: Bot对象，用于调用API
    - event: 消息事件对象，包含消息相关信息
    - matcher: Matcher对象，用于控制事件处理流程

    返回值: 无
    """
    # 如果配置中未启用调试模式，跳过后续处理
    if not config_manager.config.enable:
        matcher.skip()
    # 如果不是管理员用户，直接返回
    if event.user_id not in config_manager.config.admins:
        return
    global debug
    # 根据当前调试模式状态，开启或关闭调试模式，并发送通知
    if debug:
        debug = False
        await debug_switch.finish(
            "已关闭调试模式（该模式适用于开发者，如果你作为普通用户使用，请关闭调试模式）"
        )
    else:
        debug = True
        await debug_switch.finish(
            "已开启调试模式（该模式适用于开发者，如果你作为普通用户使用，请关闭调试模式）"
        )


# 当有消息撤回时触发处理函数
@recall.handle()
async def _(bot: Bot, event: GroupRecallNoticeEvent, matcher: Matcher):
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


# 定义聊天功能菜单的初始消息内容，包含各种命令及其描述
menu_msg = "聊天功能菜单:\n/聊天菜单 唤出菜单 \n/del_memory 丢失这个群/聊天的记忆 \n/enable 在群聊启用聊天 \n/disable 在群聊里关闭聊天\n/prompt <arg> [text] 设置聊群自定义补充prompt（--(show) 展示当前提示词，--(clear) 清空当前prompt，--(set) [文字]则设置提示词，e.g.:/prompt --(show)）,/prompt --(set) [text]。）/sessions指令帮助：\nset：覆盖当前会话为指定编号的会话\ndel：删除指定编号的会话\narchive：归档当前会话\nclear：清空所有会话\nPreset帮助：\n/presets 列出所有读取到的模型预设\n/set_preset 或 /设置预设 或 /设置模型预设  <预设名> 设置当前使用的预设"


# 处理菜单命令的函数
@menu.handle()
async def _(event: MessageEvent, matcher: Matcher):
    # 声明全局变量，用于访问和修改自定义菜单、默认菜单消息以及配置信息
    global custom_menu, menu_msg, config

    # 检查聊天功能是否已启用，未启用则跳过处理
    if not config_manager.config.enable:
        matcher.skip()

    # 初始化消息内容为默认菜单消息
    msg = menu_msg

    # 遍历自定义菜单项，添加到消息内容中
    for menus in custom_menu:
        msg += f"\n{menus['cmd']} {menus['describe']}"

    # 根据配置信息，添加群聊或私聊聊天可用性的提示信息
    msg += f"\n{'群内可以at我与我聊天，' if config_manager.config.enable_group_chat else '未启用群内聊天，'}{'在私聊可以直接聊天。' if config_manager.config.enable_group_chat else '未启用私聊聊天'}\nPowered by Suggar chat plugin"

    # 发送最终的消息内容
    await menu.send(msg)


@poke.handle()
async def _(event: PokeNotifyEvent, bot: Bot, matcher: Matcher):
    """
    处理戳一戳事件的异步函数。

    参数:
    - event: 戳一戳通知事件对象。
    - bot: 机器人对象。
    - matcher: 匹配器对象，用于控制事件处理流程。

    此函数主要根据配置信息和事件类型，响应戳一戳事件，并发送预定义的消息。
    """
    # 声明全局变量
    global debug

    # 检查配置，如果机器人未启用，则跳过处理
    if not config_manager.config.enable:
        matcher.skip()

    # 如果配置中未开启戳一戳回复，则直接返回
    if not config_manager.config.poke_reply:
        poke.skip()
        return

    # 如果事件的目标ID不是机器人自身，则直接返回
    if event.target_id != event.self_id:
        return

    try:
        # 判断事件是否发生在群聊中
        if event.group_id is not None:
            Group_Data = get_memory_data(event)
            i = Group_Data
            # 如果群聊ID匹配且群聊功能开启，则处理事件
            if i["enable"]:
                # 获取用户昵称
                user_name = (
                    await bot.get_group_member_info(
                        group_id=event.group_id, user_id=event.user_id
                    )
                )["nickname"]
                # 构建发送的消息内容
                send_messages = [
                    {"role": "system", "content": f"{config_manager.group_train}"},
                    {
                        "role": "user",
                        "content": f"\\（戳一戳消息\\){user_name} (QQ:{event.user_id}) 戳了戳你",
                    },
                ]
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
                # 初始化响应内容和调试信息
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
                # 如果调试模式开启，发送调试信息给管理员
                if debug:
                    await send_to_admin(
                        f"POKEMSG{event.group_id}/{event.user_id}\n {send_messages}"
                    )
                # 构建最终消息并发送
                message = (
                    MessageSegment.at(user_id=event.user_id)
                    + MessageSegment.text(" ")
                    + MessageSegment.text(response)
                )

                if not config_manager.config.nature_chat_style:
                    await poke.send(message)
                else:
                    if response_list := split_message_into_chats(response):
                        # 将@用户添加到第一条消息
                        first_message = (
                            MessageSegment.at(event.user_id)
                            + MessageSegment.text(" ")
                            + response_list[0]
                        )
                        await poke.send(first_message)

                        # 发送剩余消息并保持原有延迟逻辑
                        for message in response_list[1:]:
                            await poke.send(message)
                            await asyncio.sleep(
                                random.randint(1, 3)
                                + len(message) // random.randint(80, 100)
                            )

        else:
            name = get_friend_info(event.user_id)
            send_messages = [
                {"role": "system", "content": f"{config_manager.private_train}"},
                {
                    "role": "user",
                    "content": f" \\（戳一戳消息\\) {name}(QQ:{event.user_id}) 戳了戳你",
                },
            ]
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
            if debug:
                await send_to_admin(f"POKEMSG {send_messages}")
            message = MessageSegment.text(response)

            if not config_manager.config.nature_chat_style:
                await poke.send(message)
            else:
                response_list = split_message_into_chats(response)
                # await poke.send(MessageSegment.at(event.user_id))
                for message in response_list:
                    await poke.send(message)
                    await asyncio.sleep(
                            random.randint(1, 3)
                            + len(message) // random.randint(80, 100)
                    )

    except Exception:
        # 异常处理，记录错误信息并发送给管理员
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


@disable.handle()
async def _(bot: Bot, event: GroupMessageEvent, matcher: Matcher):
    """
    处理禁用聊天功能的异步函数。

    当接收到群消息事件时，检查当前配置是否允许执行禁用操作，如果不允许则跳过处理。
    检查发送消息的成员是否为普通成员且不在管理员列表中，如果是则发送提示消息并返回。
    如果成员有权限，记录日志并更新记忆中的数据结构以禁用聊天功能，然后发送确认消息。

    参数:
    - bot: Bot对象，用于调用机器人API。
    - event: GroupMessageEvent对象，包含群消息事件的相关信息。
    - matcher: Matcher对象，用于控制事件处理流程。

    返回: 无
    """
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
    """
    处理启用聊天功能的命令。

    该函数检查当前配置是否允许启用聊天功能，如果允许则检查发送命令的用户是否为管理员。
    如果用户是普通成员且不在管理员列表中，则发送提示信息并返回。
    如果用户有权限，且当前聊天功能已启用，则发送“聊天启用”的消息。
    如果聊天功能未启用，则启用聊天功能并发送“聊天启用”的消息。

    参数:
    - bot: Bot对象，用于调用API。
    - event: GroupMessageEvent对象，包含事件相关的信息。
    - matcher: Matcher对象，用于控制事件的处理流程。
    """
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
    """
    处理删除记忆指令的异步函数。

    参数:
    - bot: Bot对象，用于与机器人交互。
    - event: MessageEvent对象，包含事件的所有信息。
    - matcher: Matcher对象，用于控制事件的处理流程。

    此函数主要用于处理来自群聊或私聊的消息事件，根据用户权限删除机器人记忆中的上下文信息。
    """

    # 声明全局变量
    global admins, config

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
    global running_messages, session_clear_group, session_clear_user
    """
    处理聊天事件的主函数。

    参数:
    - event: MessageEvent - 消息事件对象，包含消息的相关信息。
    - matcher: Matcher - 用于控制事件处理流程的对象。
    - bot: Bot - 机器人对象，用于调用机器人相关API。

    此函数负责根据配置和消息类型处理不同的聊天消息，包括群聊和私聊消息的处理。
    """
    global debug
    # 检查配置，如果未启用则跳过处理
    if not config_manager.config.enable:
        matcher.skip()

    memory_lenth_limit = config_manager.config.memory_lenth_limit
    Date = get_current_datetime_timestamp()
    bot = nonebot.get_bot()
    global group_train, private_train

    content = ""
    logger.info(event.get_message())
    # 如果消息以“/”开头，则跳过处理
    if event.message.extract_plain_text().strip().startswith("/"):
        matcher.skip()
        return

    # 如果消息为“菜单”，则发送菜单消息并结束处理
    if event.message.extract_plain_text().startswith("菜单"):
        await matcher.finish(menu_msg)
        return

    Group_Data = get_memory_data(event)
    Private_Data = get_memory_data(event)

    # 根据消息类型处理消息
    if event.get_message():
        try:
            if isinstance(event, GroupMessageEvent):
                if not config_manager.config.enable_group_chat:
                    matcher.skip()
                data = Group_Data
                if data["enable"]:
                    if data.get("sessions") is None:
                        data["sessions"] = []
                    if data.get("timestamp") is None:
                        data["timestamp"] = time.time()
                    if config_manager.config.session_control:
                        for session in session_clear_group:
                            if session["id"] == event.group_id:
                                if not event.reply:
                                    session_clear_group.remove(session)
                                break
                        if (time.time() - data["timestamp"]) >= (
                            config_manager.config.session_control_time * 60
                        ):
                            data["sessions"].append(
                                {
                                    "messages": data["memory"]["messages"],
                                    "time": time.time(),
                                }
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
                            session_clear_group.append(
                                {
                                    "id": event.group_id,
                                    "message_id": chated["message_id"],
                                    "timestamp": time.time(),
                                }
                            )
                            return
                        elif event.reply:
                            if session["id"] == event.group_id and "继续" in event.reply.message.extract_plain_text():
                                try:
                                    if time.time() - session["timestamp"] < 100:
                                        await bot.delete_msg(
                                            message_id=session["message_id"]
                                        )
                                except Exception:
                                    pass
                                session_clear_group.remove(session)
                                data["memory"]["messages"] = data["sessions"][
                                    len(data["sessions"]) - 1
                                ]
                                data["sessions"].remove(
                                    data["sessions"][len(data["sessions"]) - 1]
                                )
                                await chat.send("让我们继续聊天吧～")

                    group_id = event.group_id
                    user_id = event.user_id
                    content = ""
                    user_name = (
                        await bot.get_group_member_info(
                            group_id=group_id, user_id=user_id
                        )
                    )["nickname"]
                    content = await synthesize_message(event.get_message(), bot)
                    if content.strip() == "":
                        content = ""
                    role = await bot.get_group_member_info(
                        group_id=group_id, user_id=user_id
                    )

                    if role["role"] == "admin":
                        role = "群管理员"
                    elif role["role"] == "owner":
                        role = "群主"
                    elif role["role"] == "member":
                        role = "普通成员"
                    logger.debug(f"{Date}{user_name}（{user_id}）说:{content}")
                    reply = "（（（引用的消息）））：\n"
                    if event.reply:
                        dt_object = datetime.fromtimestamp(event.reply.time)
                        weekday = dt_object.strftime("%A")
                        # 格式化输出结果
                        try:
                            rl = await bot.get_group_member_info(
                                group_id=group_id, user_id=event.reply.sender.user_id
                            )

                            if rl["rl"] == "admin":
                                rl = "群管理员"
                            elif rl["rl"] == "owner":
                                rl = "群主"
                            elif rl["rl"] == "member":
                                rl = "普通成员"
                            elif event.reply.sender.user_id == event.self_id:
                                rl = "自己"
                        except Exception:
                            if event.reply.sender.user_id == event.self_id:
                                rl = "自己"
                            else:
                                rl = "[获取身份失败]"
                        formatted_time = dt_object.strftime("%Y-%m-%d %I:%M:%S %p")
                        DT = f"{formatted_time} {weekday} [{rl}]{event.reply.sender.nickname}（QQ:{event.reply.sender.user_id}）说："
                        reply += DT
                        reply += await synthesize_message(event.reply.message, bot)
                        if config_manager.config.parse_segments:
                            content += str(reply)
                        else:
                            content += event.reply.message.extract_plain_text()
                        logger.debug(reply)
                        logger.debug(
                            f"[{role}][{Date}][{user_name}（{user_id}）]说:{content}"
                        )

                    data["memory"]["messages"].append(
                        {
                            "role": "user",
                            "content": f"[{role}][{Date}][{user_name}（{user_id}）]说:{content if config_manager.config.parse_segments else event.message.extract_plain_text()}",
                        }
                    )
                    while (len(data["memory"]["messages"]) > memory_lenth_limit) or (
                        data["memory"]["messages"][0]["role"] != "user"
                    ):
                        del data["memory"]["messages"][0]
                    if config_manager.config.enable_tokens_limit:
                        full_string = ""
                        memory_l = [
                            config_manager.group_train.copy(),
                            *data["memory"]["messages"].copy(),
                        ]
                        for st in memory_l:
                            full_string += st["content"]
                        tokens = hybrid_token_count(
                            full_string, config_manager.config.tokens_count_mode
                        )
                        logger.debug(f"tokens:{tokens}")
                        logger.debug(
                            f"tokens_limit:{config_manager.config.session_max_tokens}"
                        )
                        while tokens > config_manager.config.session_max_tokens:
                            del data["memory"]["messages"][0]
                            full_string = ""
                            for st in [
                                config_manager.group_train.copy(),
                                *data["memory"]["messages"].copy(),
                            ]:
                                full_string += st["content"]
                            tokens = hybrid_token_count(
                                full_string, config_manager.config.tokens_count_mode
                            )

                    send_messages = []
                    send_messages = data["memory"]["messages"].copy()
                    train = config_manager.group_train.copy()

                    train["content"] += (
                        f"\n以下是一些补充内容，如果与上面任何一条有冲突请忽略。\n{data.get('prompt', '无')}"
                    )
                    send_messages.insert(0, train)
                    try:
                        if config_manager.config.matcher_function:
                            _matcher = SuggarMatcher(
                                event_type=EventType().before_chat()
                            )
                            # todo send_messages传参改为data['memory']['messages']
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
                        debug_response = response
                        message = MessageSegment.reply(
                            event.message_id
                        ) + MessageSegment.text(response)

                        if debug:
                            await send_to_admin(
                                f"{event.group_id}/{event.user_id}\n{event.message.extract_plain_text()}\n{type(event)}\nRESPONSE:\n{response!s}\nraw:{debug_response}"
                            )
                            logger.debug(data["memory"]["messages"])
                            logger.debug(str(response))
                            await send_to_admin(f"response:{response}")

                        data["memory"]["messages"].append(
                            {"role": "assistant", "content": str(response)}
                        )

                        if not config_manager.config.nature_chat_style:
                            await chat.send(message)
                        else:
                            response_list = split_message_into_chats(response)
                            if response_list:  # 确保消息列表非空
                                # 将@用户添加到第一条消息
                                first_message = (
                                    MessageSegment.at(event.user_id)
                                    + MessageSegment.text(" ")
                                    + response_list[0]
                                )
                                await chat.send(first_message)

                                # 发送剩余消息并保持原有延迟逻辑
                                for message in response_list[1:]:
                                    await chat.send(message)
                                    await asyncio.sleep(
                                        random.randint(1, 3)
                                        + int(len(message) / random.randint(80, 100))
                                    )
                    except NoneBotException as e:
                        raise e
                    except Exception:
                        await chat.send("出错了，稍后试试（错误已反馈")

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
                    finally:
                        write_memory_data(event, data)

                else:
                    await chat.send("聊天没有启用")
                    return
            else:
                if not config_manager.config.enable_private_chat:
                    matcher.skip()
                data = Private_Data
                if data.get("sessions") is None:
                    data["sessions"] = []
                if data.get("timestamp") is None:
                    data["timestamp"] = time.time()
                if config_manager.config.session_control:
                    for session in session_clear_user:
                        if session["id"] == event.user_id:
                            if not event.reply:
                                session_clear_user.remove(session)
                            break
                    if (time.time() - data["timestamp"]) >= (
                        config_manager.config.session_control_time * 60
                    ):
                        data["sessions"].append(
                            {
                                "messages": data["memory"]["messages"],
                                "time": time.time(),
                            }
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
                        session_clear_user.append(
                            {
                                "id": event.user_id,
                                "message_id": chated["message_id"],
                                "timestamp": time.time(),
                            }
                        )
                        return
                    elif event.reply:
                        if session["id"] == event.user_id and "继续" in event.reply.message.extract_plain_text():
                            try:
                                if time.time() - session["timestamp"] < 100:
                                    await bot.delete_msg(
                                        message_id=session["message_id"]
                                    )
                            except Exception:
                                pass
                            session_clear_user.remove(session)
                            data["memory"]["messages"] = data["sessions"][
                                len(data["sessions"]) - 1
                            ]
                            data["sessions"].remove(
                                data["sessions"][len(data["sessions"]) - 1]
                            )
                            await chat.send("让我们继续聊天吧～")
                if data["id"] == event.user_id:
                    content = ""
                    rl = ""
                    content = await synthesize_message(event.get_message(), bot)
                    if content.strip() == "":
                        content = ""
                    logger.debug(f"{content}")
                    reply = "（（（引用的消息）））：\n"
                    if event.reply:
                        dt_object = datetime.fromtimestamp(event.reply.time)
                        weekday = dt_object.strftime("%A")
                        # 格式化输出结果

                        formatted_time = dt_object.strftime("%Y-%m-%d %I:%M:%S %p")
                        DT = f"{formatted_time} {weekday} {rl} {event.reply.sender.nickname}（QQ:{event.reply.sender.user_id}）说："
                        reply += DT
                        reply += await synthesize_message(event.reply.message, bot)
                        if config_manager.config.parse_segments:
                            content += str(reply)
                        else:
                            content += event.reply.message.extract_plain_text()
                        logger.debug(reply)

                    data["memory"]["messages"].append(
                        {
                            "role": "user",
                            "content": f"{Date}{await get_friend_info(event.user_id)}（{event.user_id}）： {str(content) if config_manager.config.parse_segments else event.message.extract_plain_text()}",
                        }
                    )
                    while (len(data["memory"]["messages"]) > memory_lenth_limit) or (
                        data["memory"]["messages"][0]["role"] != "user"
                    ):
                        del data["memory"]["messages"][0]
                    if config_manager.config.enable_tokens_limit:
                        full_string = ""
                        memory_l = [
                            config_manager.private_train.copy(),
                            *data["memory"]["messages"].copy(),
                        ]
                        for st in memory_l:
                            full_string += st["content"]
                        tokens = hybrid_token_count(
                            full_string, config_manager.config.tokens_count_mode
                        )
                        logger.debug(f"tokens:{tokens}")
                        logger.debug(
                            f"tokens_limit:{config_manager.config.session_max_tokens}"
                        )
                        while tokens > config_manager.config.session_max_tokens:
                            del data["memory"]["messages"][0]
                            full_string = ""
                            for st in [
                                config_manager.private_train.copy(),
                                *data["memory"]["messages"].copy(),
                            ]:
                                full_string += st["content"]
                            tokens = hybrid_token_count(
                                full_string, config_manager.config.tokens_count_mode
                            )
                    send_messages = []
                    send_messages = data["memory"]["messages"].copy()
                    send_messages.insert(0, config_manager.private_train)
                    try:
                        if config_manager.config.matcher_function:
                            _matcher = SuggarMatcher(
                                event_type=EventType().before_chat()
                            )
                            # todo send_messages传参改为data['memory']['messages']
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
                        debug_response = response
                        if debug:
                            await send_to_admin(
                                f"{event.user_id}\n{type(event)}\n{event.message.extract_plain_text()}\nRESPONSE:\n{response!s}\nraw:{debug_response}"
                            )
                        message = MessageSegment.text(response)

                        if debug:
                            logger.debug(data["memory"]["messages"])
                            logger.debug(str(response))
                            await send_to_admin(f"response:{response}")

                        data["memory"]["messages"].append(
                            {"role": "assistant", "content": str(response)}
                        )

                        if not config_manager.config.nature_chat_style:
                            await chat.send(message)
                        else:
                            # await chat.send(MessageSegment.at(event.user_id))
                            response_list = split_message_into_chats(response)
                            for response in response_list:
                                await chat.send(response)
                                await asyncio.sleep(
                                    random.randint(1, 3)
                                    + int(len(message) / random.randint(80, 100))
                                )
                    except NoneBotException as e:
                        raise e
                    except Exception:
                        exc_type, exc_value, exc_traceback = sys.exc_info()
                        await chat.send("出错了稍后试试（错误已反馈")
                        logger.error(
                            f"Exception type: {exc_type.__name__}"
                            if exc_type
                            else "Exception type: None"
                        )
                        logger.error(f"Exception message: {exc_value!s}")
                        import traceback

                        await send_to_admin(f"出错了！{exc_value},\n{exc_type!s}")
                        await send_to_admin(f"{traceback.format_exc()} ")
                        logger.error(
                            f"Detailed exception info:\n{''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))}"
                        )
                    finally:
                        write_memory_data(event, data)
        except NoneBotException as e:
            raise e
        except Exception:
            await chat.send("出错了稍后试试吧（错误已反馈 ")

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
