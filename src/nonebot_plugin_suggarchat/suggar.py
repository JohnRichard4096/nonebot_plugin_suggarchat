from nonebot import on_command,on_notice,on_message,get_driver
from nonebot_plugin_uninfo import Uninfo
from nonebot.adapters import Event
import nonebot.adapters
from nonebot.rule import to_me
from nonebot.adapters import Message
from nonebot.params import CommandArg
from .conf import __KERNEL_VERSION__,current_directory,config_dir,main_config,custom_models_dir
from .resources import get_current_datetime_timestamp,get_config,\
     get_friend_info,get_memory_data,write_memory_data\
     ,get_models,save_config,get_group_prompt,get_private_prompt,synthesize_message
from nonebot.adapters.onebot.v11 import Message, MessageSegment, GroupMessageEvent,  \
    GroupIncreaseNoticeEvent, Bot, \
    PokeNotifyEvent,GroupRecallNoticeEvent\
    , MessageEvent
from nonebot import logger
from nonebot.matcher import Matcher
import sys
import openai
import random
from datetime import datetime  
#import aiohttp

config = get_config()
ifenable = config['enable']
random_reply = config['fake_people']
random_reply_rate = config['probability']
keyword = config['keyword']
debug = False
admins = config['admins']
custom_menu = []
private_train = get_private_prompt()
group_train = get_group_prompt()
running_messages = {}
running_messages_poke = {}

async def send_to_admin(msg:str)-> None:
    """
    异步发送消息给管理员。

    该函数会检查配置文件是否允许发送消息给管理员，以及是否配置了管理员群号。
    如果满足条件，则发送消息；否则，将记录警告日志。

    参数:
    msg (str): 要发送给管理员的消息。

    返回:
    无返回值。
    """
    global config
    # 检查是否允许发送消息给管理员
    if not config['allow_send_to_admin']: return
    # 检查管理员群号是否已配置
    if config['admin_group'] == 0:
        try:
            # 如果未配置管理员群号但尝试发送消息，抛出警告
            raise RuntimeWarning("Error!Admin group not set!")
        except Exception:
            # 记录警告日志并捕获异常信息
            logger.warning(f"Admin group hasn't set yet!So warning message\"{msg}\"wont be sent to admin group!")
            exc_type, exc_vaule, exc_tb = sys.exc_info()
            logger.exception(f"{exc_type}:{exc_vaule}")
        return
    # 获取bot实例并发送消息到管理员群
    bot: Bot = nonebot.get_bot()
    await bot.send_group_msg(group_id=config['admin_group'], message=msg)

##fakepeople rule
async def rule(event: MessageEvent, session: Uninfo, bot: Bot) -> bool:
    """
    根据配置和消息事件判断是否触发回复的规则。

    参数:
    - event: MessageEvent 类型的事件，包含消息事件的详细信息。
    - session: Uninfo 类型的会话信息，可能包含与当前会话相关的上下文或配置信息。
    - bot: Bot 类型的机器人实例，用于调用机器人相关方法。

    返回值:
    - bool 类型，表示是否触发回复的规则。
    """
    global random_reply, random_reply_rate, keyword
    # 获取消息内容并去除前后空格
    message = event.get_message()
    message_text = message.extract_plain_text().strip()
    
    # 如果不是群消息事件，则总是返回 True，表示总是回复私聊消息
    if not isinstance(event, GroupMessageEvent):
        return True
    
    # 根据配置中的 keyword 判断是否需要回复
    if keyword == "at":
        # 如果配置中的 keyword 为 "at"，则当消息是提到机器人时回复
        if event.is_tome():
            return True
    else:
        # 如果配置中的 keyword 不为 "at"，则当消息文本以 keyword 开头时回复
        if message_text.startswith(keyword):
            """开头为{keyword}必定回复"""
            return True
    
    # 如果没有开启随机回复功能，则不回复
    if not random_reply:
        return False
    else:
        # 私聊消息不进行随机回复
        if event.is_tome() and not session.group:
            """私聊过滤"""
            return False
        
        # 将 event 强制转换为 GroupMessageEvent 类型
        event: GroupMessageEvent = event
        
        # 根据随机率判断是否回复
        rand = random.randint(1, 100)
        rate = random_reply_rate
        if rand <= rate:
            return True
        
        # 获取记忆数据
        memory_data: dict = get_memory_data(event)
        
        # 合成消息内容
        content = await synthesize_message(message, bot)
        
        # 获取当前时间戳
        Date = get_current_datetime_timestamp()
        
        # 获取消息发送者的角色（群成员或管理员等）
        role = (await bot.get_group_member_info(group_id=event.group_id, user_id=event.user_id))['role']
        
        # 获取消息发送者的用户ID和昵称
        user_id = event.user_id
        user_name = (await bot.get_group_member_info(group_id=event.group_id, user_id=event.user_id))['nickname'] or (await bot.get_stranger_info(user_id=event.user_id))['nickname']
        
        # 构造消息记录格式
        content_message = f"[{role}][{Date}][{user_name}（{user_id}）]说:{content}"
        
        # 将消息记录添加到记忆数据中
        memory_data["memory"]["messages"].append(content_message)
        
        # 更新记忆数据
        write_memory_data(event, memory_data)
        
        # 不回复
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
    user_role = await bot.get_group_member_info(group_id=event.group_id, user_id=event.user_id)
    # 提取成员在群组中的角色
    user_role = user_role.get("role")
    # 判断成员角色是否为"member"（普通成员）
    if user_role == "member":
        return True
    return False


async def get_chat(messages:list)->str:
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
    global config, ifenable,debug
    # 从配置中获取最大token数量
    max_tokens = config['max_tokens']
    
    # 根据配置中的预设值，选择不同的API密钥和基础URL
    if config['preset'] == "__main__":
        # 如果是主配置，直接使用配置文件中的设置
        base_url = config['open_ai_base_url']
        key = config['open_ai_api_key']
        model = config['model']
    else:
        # 如果是其他预设，从模型列表中查找匹配的设置
        models = get_models()
        for i in models:
            if i['name'] == config['preset']:
                base_url = i['base_url']
                key = i['api_key']
                model = i['model']
                break
        else:
            # 如果未找到匹配的预设，记录错误并重置预设为主配置文件
            logger.error(f"Preset {config['preset']} not found")
            logger.info("Found：Main config，Model："+config['model'])
            config['preset'] = "__main__"
            key = config['open_ai_api_key']
            model = config['model']
            base_url = config['open_ai_base_url']
            # 保存更新后的配置
            save_config(config)
    
    # 记录日志，开始获取对话
    logger.debug(f"Start to get response with model {model}")
    logger.debug(f"Preset：{config['preset']}")
    logger.debug(f"Key：{key[:10]}...")
    logger.debug(f"API base_url：{base_url}")

    client = openai.AsyncOpenAI( base_url=base_url, api_key=key)
        # 创建聊天完成请求
    completion = await client.chat.completions.create(model=model, messages=messages, max_tokens=max_tokens, stream=config['stream'])
    response = ""
    if config['stream']:
        # 流式接收响应并构建最终的聊天文本
        async for chunk in completion:
            try:
                response += chunk.choices[0].delta.content
                if debug:
                    logger.debug(chunk.choices[0].delta.content)
            except IndexError:
                break
        # 记录生成的响应日志
    else:
        if debug:
            logger.debug(response)
        response = completion.choices[0].message.content
    return response

#创建响应器实例
#fake_people = on_notice(block=False)
add_notice = on_notice(block=False)
menu = on_command("聊天菜单",block=True,aliases={"chat_menu"},priority=10)
chat = on_message(block=False,priority=11,rule=rule)#不再在此处判断是否触发,转到line68
del_memory = on_command("del_memory",aliases={"失忆","删除记忆","删除历史消息","删除回忆"},block=True,priority=10)
enable = on_command("enable_chat",aliases={"启用聊天"},block=True,priority=10)
disable = on_command("disable_chat",aliases={"禁用聊天"},block=True,priority=10)
poke = on_notice(priority=10,block=True)
debug_switch = on_command("debug",priority=10,block=True)
debug_handle = on_message(rule=to_me(),priority=10,block=False)
recall = on_notice()
prompt = on_command("prompt",priority=10,block=True)
presets = on_command("presets",priority=10,block=True)
set_preset = on_command("set_preset",aliases={"设置预设","设置模型预设"},priority=10,block=True)
del_all_memory = on_command("del_all_memory",priority=10,block=True)
@del_all_memory.handle()
async def del_all_memory_handle(bot:Bot,event:MessageEvent):
    global config
    if not event.user_id in config["admins"]:
        await del_all_memory.finish("你没有权限执行此操作")
    
# 处理设置预设的函数
@set_preset.handle()
async def _(bot: Bot, event: MessageEvent, args: Message = CommandArg()):
    # 声明全局变量
    global admins,config,ifenable
    
    # 检查插件是否启用
    if not ifenable:
        set_preset.skip()
    
    # 检查用户是否为管理员
    if not event.user_id in admins:
        await set_preset.finish("只有管理员才能设置预设。")
    
    # 提取命令参数
    arg = args.extract_plain_text().strip()
    
    # 如果参数不为空
    if not arg == "":
        # 获取模型列表
        models = get_models()
        
        # 遍历模型列表
        for i in models:
            # 如果模型名称与参数匹配
            if i['name'] == arg:
                # 设置预设并保存配置
                config['preset'] = i['name']
                save_config(config)
                # 回复设置成功
                await set_preset.finish(f"已设置预设为：{i['name']}，模型：{i['model']}")
                break
        else:
            # 如果未找到预设，提示用户
            set_preset.finish("未找到预设，请输入/presets查看预设列表。")
    else:
        # 如果参数为空，重置预设为默认
        config['preset'] = "__main__"
        save_config(config)
        # 回复重置成功
        await set_preset.finish("已重置预设为：主配置文件，模型："+config['model'])
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
    # 声明全局变量
    global admins, config, ifenable
    
    # 检查功能是否已启用，未启用则跳过处理
    if not ifenable:
        presets.skip()
    
    # 检查用户是否为管理员，非管理员则发送消息并结束处理
    if not event.user_id in admins:
        await presets.finish("只有管理员才能查看模型预设。")
    
    # 获取模型列表
    models = get_models()
    
    # 构建消息字符串，包含当前模型预设信息
    msg = f"模型预设:\n当前：{'主配置文件' if config['preset'] == '__main__' else config['preset']}\n主配置文件：{config['model']}"
    
    # 遍历模型列表，添加每个预设的名称和模型到消息字符串
    for i in models:
        msg += f"\n预设名称：{i['name']}，模型：{i['model']}"
    
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
    global config
    # 检查是否启用prompt功能，未启用则跳过处理
    if not config['enable']:
        prompt.skip()
    # 检查是否允许自定义prompt，不允许则结束处理
    if not config['allow_custom_prompt']:
        await prompt.finish("当前不允许自定义prompt。")
    
    global admins
    # 检查用户是否为群成员且非管理员，是则结束处理
    if await is_member(event, bot) and not event.user_id in admins:
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
        await prompt.send("请输入prompt或参数（--(show) 展示当前提示词，--(clear) 清空当前prompt，--(set) [文字]则设置提示词，e.g.:/prompt --(show)）,/prompt --(set) [text]。）")
        return
    
    # 根据用户输入的不同命令进行相应的处理
    if arg.startswith("--(show)"):
        await prompt.send(f"Prompt:\n{data.get('prompt','未设置prompt')}")
        return
    elif arg.startswith("--(clear)"):
        data['prompt'] = ""
        await prompt.send("prompt已清空。")
    elif arg.startswith("--(set)"):
        arg = arg.replace("--(set)","").strip()
        data['prompt'] = arg
        await prompt.send(f"prompt已设置为：\n{arg}")
    else:
        await prompt.send("请输入prompt或参数（--(show) 展示当前提示词，--(clear) 清空当前prompt，--(set) [文字]则设置提示词，e.g.:/prompt --(show)）,/prompt --(set) [text]。")
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
    global config
    # 检查全局配置，如果未启用，则跳过处理
    if not config['enable']:
        add_notice.skip()
    # 检查配置，如果不发送被邀请后的消息，则直接返回
    if not config['send_msg_after_be_invited']:
        return
    # 如果事件的用户ID与机器人自身ID相同，表示机器人被邀请加入群聊
    if event.user_id == event.self_id:
        # 发送配置中的群聊添加消息
        await add_notice.send(config['group_added_msg'])
        return

# 处理调试模式开关的函数
@debug_switch.handle()
async def _ (bot:Bot,event:MessageEvent,matcher:Matcher):
    """
    根据用户权限开启或关闭调试模式。
    
    参数:
    - bot: Bot对象，用于调用API
    - event: 消息事件对象，包含消息相关信息
    - matcher: Matcher对象，用于控制事件处理流程
    
    返回值: 无
    """
    global admins,config
    # 如果配置中未启用调试模式，跳过后续处理
    if not config['enable']:matcher.skip()
    # 如果不是管理员用户，直接返回
    if not event.user_id in admins:
        return
    global debug
    # 根据当前调试模式状态，开启或关闭调试模式，并发送通知
    if debug:
        debug = False
        await debug_switch.finish("已关闭调试模式（该模式适用于开发者，如果你作为普通用户使用，请关闭调试模式）")
    else:
        debug = True
        await debug_switch.finish("已开启调试模式（该模式适用于开发者，如果你作为普通用户使用，请关闭调试模式）")


# 当有消息撤回时触发处理函数
@recall.handle()
async def _(bot:Bot,event:GroupRecallNoticeEvent,matcher:Matcher):
    # 声明全局变量config，用于访问配置信息
    global config
    # 检查是否启用了插件功能，未启用则跳过后续处理
    if not config['enable']:matcher.skip()
    # 通过随机数决定是否响应，增加趣味性和减少响应频率
    if not random.randint(1,3) == 2:
        return
    # 检查配置中是否允许在删除自己的消息后发言，不允许则直接返回
    if not config['say_after_self_msg_be_deleted']:return
    # 从配置中获取删除消息后可能的回复内容
    recallmsg = config['after_deleted_say_what']
    # 判断事件是否为机器人自己删除了自己的消息
    if event.user_id == event.self_id:
        # 如果是机器人自己删除了自己的消息，并且操作者也是机器人自己，则不进行回复
        if event.operator_id == event.self_id:
            return
        # 从预设的回复内容中随机选择一条发送
        await recall.send(random.choice(recallmsg))
        return





# 定义聊天功能菜单的初始消息内容，包含各种命令及其描述
menu_msg = "聊天功能菜单:\n/聊天菜单 唤出菜单 \n/del_memory 丢失这个群/聊天的记忆 \n/enable 在群聊启用聊天 \n/disable 在群聊里关闭聊天\n/prompt <arg> [text] 设置聊群自定义补充prompt（--(show) 展示当前提示词，--(clear) 清空当前prompt，--(set) [文字]则设置提示词，e.g.:/prompt --(show)）,/prompt --(set) [text]。）"

# 处理菜单命令的函数
@menu.handle()
async def _(event:MessageEvent,matcher:Matcher):
    # 声明全局变量，用于访问和修改自定义菜单、默认菜单消息以及配置信息
    global custom_menu,menu_msg,config
    
    # 检查聊天功能是否已启用，未启用则跳过处理
    if not config['enable']:
        matcher.skip()
    
    # 初始化消息内容为默认菜单消息
    msg = menu_msg
    
    # 遍历自定义菜单项，添加到消息内容中
    for menus in custom_menu:
        msg += f"\n{menus['cmd']} {menus['describe']}"
    
    # 根据配置信息，添加群聊或私聊聊天可用性的提示信息
    msg += f"\n{'群内可以at我与我聊天，' if config['enable_group_chat'] else '未启用群内聊天，'}{'在私聊可以直接聊天。' if config['enable_private_chat'] else '未启用私聊聊天'}\nPowered by Suggar chat plugin"
    
    # 发送最终的消息内容
    await menu.send(msg)

@poke.handle()
async def _(event:PokeNotifyEvent,bot:Bot,matcher:Matcher):
    """
    处理戳一戳事件的异步函数。
    
    参数:
    - event: 戳一戳通知事件对象。
    - bot: 机器人对象。
    - matcher: 匹配器对象，用于控制事件处理流程。
    
    此函数主要根据配置信息和事件类型，响应戳一戳事件，并发送预定义的消息。
    """
    # 声明全局变量，用于获取prompt和调试模式
    global private_train,group_train
    global debug,config
    
    # 检查配置，如果机器人未启用，则跳过处理
    if not config['enable']:
        matcher.skip()
    
    # 如果配置中未开启戳一戳回复，则直接返回
    if not config['poke_reply']:
        poke.skip()
        return
    
    # 获取群聊和私聊的数据
    Group_Data = get_memory_data(event)
    Private_Data = get_memory_data(event)
    
    # 如果事件的目标ID不是机器人自身，则直接返回
    if event.target_id != event.self_id:
        return

    try:
        # 判断事件是否发生在群聊中
        if event.group_id != None:
            i = Group_Data
            # 如果群聊ID匹配且群聊功能开启，则处理事件
            if i['id'] == event.group_id and i['enable']:
                # 获取用户昵称
                user_name = (await bot.get_group_member_info(group_id=event.group_id, user_id=event.user_id))['nickname'] or (await bot.get_stranger_info(user_id=event.user_id))['nickname']
                # 构建发送的消息内容
                send_messages = [
                    {"role": "system", "content": f"{group_train}"},
                    {"role": "user", "content": f"\\（戳一戳消息\\){user_name} (QQ:{event.user_id}) 戳了戳你"}
                ]
                
                # 初始化响应内容和调试信息
                response = await get_chat(send_messages)
                
                # 如果调试模式开启，发送调试信息给管理员
                if debug:
                    await send_to_admin(f"POKEMSG{event.group_id}/{event.user_id}\n {send_messages}") 
                # 构建最终消息并发送
                message = MessageSegment.at(user_id=event.user_id) +MessageSegment.text(" ")+ MessageSegment.text(response)
                i['memory']['messages'].append({"role":"assistant","content":str(response)})
                
                # 更新群聊数据
                write_memory_data(event,i)
                await poke.send(message)
        
        else:
            # 如果事件发生在私聊中，执行类似的处理流程
            i = Private_Data
            if i['id'] == event.user_id and i['enable']:
                name = get_friend_info(event.user_id)
                send_messages = [
                    {"role": "system", "content": f"{private_train}"},
                    {"role": "user", "content": f" \\（戳一戳消息\\) {name}(QQ:{event.user_id}) 戳了戳你"}
                ]
                
                response = await get_chat(send_messages)
                if debug:
                    await send_to_admin(f"POKEMSG {send_messages}") 
                    
                
                message = MessageSegment.text(response)
                i['memory']['messages'].append({"role":"assistant","content":str(response)})
                write_memory_data(event,i)
                await poke.send(message)
                
    except Exception as e:
        # 异常处理，记录错误信息并发送给管理员
        exc_type, exc_value, exc_traceback = sys.exc_info()
        logger.error(f"Exception type: {exc_type.__name__}")  
        logger.error(f"Exception message: {str(exc_value)}")  
        import traceback  
        await send_to_admin(f"出错了！{exc_value},\n{str(exc_type)}")
        await send_to_admin(f"{traceback.format_exc()}")
        
        logger.error(f"Detailed exception info:\n{''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))}")       



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
    global admins, config
    # 检查全局配置是否启用，如果未启用则跳过后续处理
    if not config['enable']:
        matcher.skip()
    
    # 获取发送消息的成员信息
    member = await bot.get_group_member_info(group_id=event.group_id, user_id=event.user_id)
    
    # 检查成员是否为普通成员且不在管理员列表中，如果是则发送提示消息并返回
    if member['role'] == "member" and event.user_id not in admins:
        await disable.send("你没有这样的力量呢～（管理员/管理员+）")
        return
    
    # 记录禁用操作的日志
    logger.debug(f"{event.group_id} disabled")
    
    # 获取并更新记忆中的数据结构
    datag = get_memory_data(event)
    if True:
        if datag['id'] == event.group_id:
            if not datag['enable']:
                await disable.send("聊天禁用")
            else:
                datag['enable'] = False
                await disable.send("聊天已经禁用")
    
    # 将更新后的数据结构写回记忆
    write_memory_data(event, datag)

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
    global admins, config
    # 检查全局配置，如果未启用则跳过后续处理
    if not config['enable']:
        matcher.skip()

    # 获取发送命令的用户在群中的角色信息
    member = await bot.get_group_member_info(group_id=event.group_id, user_id=event.user_id)
    # 如果用户是普通成员且不在管理员列表中，则发送提示信息并返回
    if member['role'] == "member" and event.user_id not in admins:
        await enable.send("你没有这样的力量呢～（管理员/管理员+）")
        return

    # 记录日志
    logger.debug(f"{event.group_id}enabled")
    # 获取记忆中的数据
    datag = get_memory_data(event)
    # 检查记忆数据是否与当前群组相关
    if True:
        if datag['id'] == event.group_id:
            # 如果聊天功能已启用，则发送提示信息
            if datag['enable']:
                await enable.send("聊天启用")
            else:
                # 如果聊天功能未启用，则启用并发送提示信息
                datag['enable'] = True
                await enable.send("聊天启用")
    # 更新记忆数据
    write_memory_data(event, datag)

   
@del_memory.handle()
async def _(bot:Bot,event:MessageEvent,matcher:Matcher):
    """
    处理删除记忆指令的异步函数。
    
    参数:
    - bot: Bot对象，用于与机器人交互。
    - event: MessageEvent对象，包含事件的所有信息。
    - matcher: Matcher对象，用于控制事件的处理流程。
    
    此函数主要用于处理来自群聊或私聊的消息事件，根据用户权限删除机器人记忆中的上下文信息。
    """
    
    # 声明全局变量
    global admins,config
    
    # 检查配置以确定是否启用功能
    if not config['enable']:
        matcher.skip()
    
    # 判断事件是否来自群聊
    if isinstance(event,GroupMessageEvent):
        # 获取群成员信息
        member = await bot.get_group_member_info(group_id=event.group_id,user_id=event.user_id)
        
        # 检查用户权限，非管理员且不在管理员列表中的用户将被拒绝
        if member['role'] == "member" and not event.user_id in admins:
            await del_memory.send("你没有这样的力量（管理员/管理员+）")
            return
        
        # 获取群聊记忆数据
        GData = get_memory_data(event)
        
        # 清除群聊上下文
        if True:
            if GData['id'] == event.group_id:
                GData['memory']['messages'] = []
                await del_memory.send("上下文已清除")
                write_memory_data(event,GData)
                logger.debug(f"{event.group_id}Memory deleted")
                
    else:
        # 获取私聊记忆数据
        FData = get_memory_data(event)
        
        # 清除私聊上下文
        if FData['id'] == event.user_id:
            FData['memory']['messages'] = []
            await del_memory.send("上下文已清除")
            logger.debug(f"{event.user_id}Memory deleted")
            write_memory_data(event,FData)
       
@get_driver().on_bot_connect
async def onConnect():
    from .conf import group_memory,private_memory
    from pathlib import Path
    from .conf import init
    bot:Bot = nonebot.get_bot()
    logger.info(f"Bot {bot.self_id} connected")
    init(bot)
    logger.info(f"Config dir：{config_dir}") 
    logger.info(f"Main config location：{main_config}")
    logger.info(f"Group memory data location：{group_memory}")
    logger.info(f"Private memory data location：{private_memory}")
    logger.info(f"Model presets dir：{custom_models_dir}")
    save_config(get_config(no_base_prompt=True))
    
@get_driver().on_startup
async def onEnable():
    logger.info(f"""
NONEBOT PLUGIN SUGGARCHAT
{__KERNEL_VERSION__}
""")
 
    logger.info("Start successfully!Waitting for bot connection...")
    


@chat.handle()
async def _(event:MessageEvent, matcher:Matcher, bot:Bot):
    global running_messages
    """
    处理聊天事件的主函数。
    
    参数:
    - event: MessageEvent - 消息事件对象，包含消息的相关信息。
    - matcher: Matcher - 用于控制事件处理流程的对象。
    - bot: Bot - 机器人对象，用于调用机器人相关API。
    
    此函数负责根据配置和消息类型处理不同的聊天消息，包括群聊和私聊消息的处理。
    """
    global debug, config
    # 检查配置，如果未启用则跳过处理
    if not config['enable']:
        matcher.skip()
    
    memory_lenth_limit = config['memory_lenth_limit']
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
        if isinstance(event,GroupMessageEvent):
                if not config['enable_group_chat']:matcher.skip()
                datag = Group_Data
                if datag['id'] == event.group_id:
                    if not datag['enable']:
                        await chat.send( "聊天没有启用，快去找管理员吧！")
                        chat.skip()
                        return
                    
                    group_id = event.group_id
                    user_id = event.user_id
                    content = ""
                    user_name = (await bot.get_group_member_info(group_id=group_id, user_id=user_id))['card'] or (await bot.get_stranger_info(user_id=user_id))['nickname']
                    content = await synthesize_message(event.get_message())
                    if content.strip() == "":
                         content = ""
                    role = await bot.get_group_member_info(group_id=group_id, user_id=user_id)
                    
                    if role['role'] == "admin":
                         role = "群管理员"
                    elif role['role'] == "owner":
                         role = "群主"
                    elif role['role'] == "member":
                         role = "普通成员"
                    logger.debug(f"{Date}{user_name}（{user_id}）说:{content}")
                    reply = "（（（引用的消息）））：\n"
                    if event.reply:
                         dt_object = datetime.fromtimestamp(event.reply.time)  
                         weekday = dt_object.strftime('%A')  
                        # 格式化输出结果  
                         try:
                          rl = await bot.get_group_member_info(group_id=group_id, user_id=event.reply.sender.user_id)
                          
                          if rl['rl'] == "admin":
                            rl = "群管理员"
                          elif rl['rl'] == "owner":
                            rl = "群主"
                          elif rl['rl'] == "member":
                            rl = "普通成员"
                          elif event.reply.sender.user_id==event.self_id:
                            rl = "自己"
                         except:
                            if event.reply.sender.user_id==event.self_id:
                                rl = "自己"
                            else:
                                rl = "[获取身份失败]"
                         formatted_time = dt_object.strftime('%Y-%m-%d %I:%M:%S %p') 
                         DT = f"{formatted_time} {weekday} [{rl}]{event.reply.sender.nickname}（QQ:{event.reply.sender.user_id}）说：" 
                         reply += DT
                         reply += await synthesize_message(event.reply.message)
                         if config['parse_segments']:
                                content += str(reply)
                         else:
                                content += event.reply.message.extract_plain_text()
                         logger.debug(reply)
                         logger.debug(f"[{role}][{Date}][{user_name}（{user_id}）]说:{content}")
    
                    datag['memory']['messages'].append({"role":"user","content":f"[{role}][{Date}][{user_name}（{user_id}）]说:{content if config['parse_segments'] else event.message.extract_plain_text()}" })
                    if len(datag['memory']['messages']) >memory_lenth_limit:
                        while len(datag['memory']['messages'])>memory_lenth_limit:
                            del datag['memory']['messages'][0]
                    send_messages = []
                    send_messages = datag['memory']['messages'].copy()
                    train = group_train.copy()
                    
                    train['content'] += f"\n以下是一些补充内容，如果与上面任何一条有冲突请忽略。\n{datag.get('prompt','无')}"
                    send_messages.insert(0,train)
                    try:    
                            
                            response = await get_chat(send_messages)
                            debug_response = response
                            message = MessageSegment.reply(event.message_id) + MessageSegment.text(response) 
                           
                            if debug:
                                 await send_to_admin(f"{event.group_id}/{event.user_id}\n{event.message.extract_plain_text()}\n{type(event)}\nRESPONSE:\n{str(response)}\nraw:{debug_response}")
                            if debug:
                                 logger.debug(datag['memory']['messages'])
                                 logger.debug(str(response))
                                 await send_to_admin(f"response:{response}")
                                 
                            datag['memory']['messages'].append({"role":"assistant","content":str(response)})
                            await chat.send(message)
                    
                    except Exception as e:
                        await chat.send(f"出错了，稍后试试（错误已反馈") 
                        
                        exc_type, exc_value, exc_traceback = sys.exc_info()
                        logger.error(f"Exception type: {exc_type.__name__}")  
                        logger.error(f"Exception message: {str(exc_value)}")  
                        import traceback  
                        await send_to_admin(f"出错了！{exc_value},\n{str(exc_type)}")
                        await send_to_admin(f"{traceback.format_exc()}")
                        
                        logger.error(f"Detailed exception info:\n{''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))}")      
 
            
                    write_memory_data(event,datag) 
        else:
                if not config['enable_private_chat']:matcher.skip()
                data = Private_Data
                if data['id'] == event.user_id:
                    content = ""
                    rl = ""
                    content = synthesize_message(event.get_message())
                    if content.strip() == "":
                         content = ""
                    logger.debug(f"{content}")
                    reply = "（（（引用的消息）））：\n"
                    if event.reply:
                         dt_object = datetime.fromtimestamp(event.reply.time)  
                         weekday = dt_object.strftime('%A')  
                        # 格式化输出结果  
                         
                         formatted_time = dt_object.strftime('%Y-%m-%d %I:%M:%S %p') 
                         DT = f"{formatted_time} {weekday} {rl} {event.reply.sender.nickname}（QQ:{event.reply.sender.user_id}）说：" 
                         reply += DT
                         reply+=await synthesize_message(event.reply.message)
                         if config['parse_segments']:
                            content += str(reply)
                         else:
                            content += event.reply.message.extract_plain_text()
                         logger.debug(reply)
                     
                    data['memory']['messages'].append({"role":"user","content":f"{Date}{await get_friend_info(event.user_id)}（{event.user_id}）： {str(content)if config['parse_segments'] else event.message.extract_plain_text()}" })
                    if len(data['memory']['messages']) >memory_lenth_limit:
                        while len(data['memory']['messages'])>memory_lenth_limit:
                            del data['memory']['messages'][0]
                    send_messages = []
                    send_messages = data['memory']['messages'].copy()
                    send_messages.insert(0,private_train)
                    try:    
                            response = await get_chat(send_messages)
                            debug_response = response
                            if debug:
                                 if debug:
                                    await send_to_admin(f"{event.user_id}\n{type(event)}\n{event.message.extract_plain_text()}\nRESPONSE:\n{str(response)}\nraw:{debug_response}")
                            message =  MessageSegment.text(response)
                            
                            
                            
                            if debug:
                                 logger.debug(data['memory']['messages'])
                                 logger.debug(str(response))
               
                                 await send_to_admin(f"response:{response}")
                                 
                            data['memory']['messages'].append({"role":"assistant","content":str(response)})
                            await chat.send(message)
                           
                            
                                
                    except Exception as e:
                        exc_type, exc_value, exc_traceback = sys.exc_info()
                        await chat.send(f"出错了稍后试试（错误已反馈")
                        logger.error(f"Exception type: {exc_type.__name__}")  
                        logger.error(f"Exception message: {str(exc_value)}")  
                        import traceback  
                        await send_to_admin(f"出错了！{exc_value},\n{str(exc_type)}")
                        await send_to_admin(f"{traceback.format_exc()} ")
                       
                        logger.error(f"Detailed exception info:\n{''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))}")      
              
                        write_memory_data(event,data)      
     except Exception as e:
                        await chat.send(f"出错了稍后试试吧（错误已反馈 ") 
                        
                        exc_type, exc_value, exc_traceback = sys.exc_info()
                        logger.error(f"Exception type: {exc_type.__name__}")  
                        logger.error(f"Exception message: {str(exc_value)}")  
                        import traceback  
                        await send_to_admin(f"出错了！{exc_value},\n{str(exc_type)}")
                        await send_to_admin(f"{traceback.format_exc()}")
                        logger.error(f"Detailed exception info:\n{''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))}")    
    else:pass
