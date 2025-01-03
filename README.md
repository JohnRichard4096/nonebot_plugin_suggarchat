<p align="center">
  <a href="https://github.com/JohnRichard4096/nonebot_plugin_suggarchat/"><img src="https://v2.nonebot.dev/logo.png" width="200" height="200" alt="nonebot"></a>
</p>

<div align="center">

# Nonebot Plugin Suggarchat

Plugin for the Suggar chat framework compatible with Nonebot2.


[![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?style=for-the-badge&logo=visual-studio-code&logoColor=white)](https://code.visualstudio.com/) [![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/) [![PyPi](https://img.shields.io/badge/pypi-%23ececec.svg?style=for-the-badge&logo=pypi&logoColor=1f73b7)](https://pypi.org/project/nonebot-plugin-suggarchat/)

## 描述
适用于**Nonebot2**的**Onebot V11 适配器**的LLM聊天插件

## 温馨提示

本插件全局性较高！***可能会与其他插件冲突***，请谨慎使用！

如果你想主要做聊天类型的Bot，请忽略。

在Suggar机器人实现中，暂时没有发现有冲突，如果有，请提交至Issues。

***本插件更面向于有Nonebot2基础或插件开发者以及有LLM API开发/使用经验的用户！***

不推荐直接调用**resources.py的**方法，而是通过插件API进行调用。

<hr />

传入LLM的信息格式如下，请自行在**提示词**内做好处理，便于LLM理解（这里我提供了use_base_prompt选项，如果启用了可以忽略，这个选项将自动在你的prompt前插入内容，对消息段作出解释）：


~~不会写就来QQ群问群主吧（~~
<details><summary>点击查看详细格式</summary>
可解析的消息段：文字，at，合并转发

<hr />

at+文字：
```plaintext
你好世界@Somebody
```
合并转发（暂不支持解析**嵌套的合并转发消息**）：
```
\（合并转发
[YYYY-MM-DD hh:mm:ss PM/AM][昵称(QQ号)]说：<内容>
[[YYYY-MM-DD hh:mm:ss PM/AM][昵称(QQ号)]说：<内容>]
）\
......以此类推
```

<hr />

私聊普通消息：
```plaintext
[YYYY-MM-DD weekday hh:mm:ss AM/PM]用户昵称（QQ号）：<内容>
```
私聊引用消息：
```plaintext
私聊普通消息格式+ （（（引用的消息）））：其他消息段解析后内容
```
私聊合并转发消息：
```plaintext
私聊普通消息格式+合并转发消息格式
```

<hr />

聊群普通消息：
```plaintext
[管理员/群主/自己/群员][YYYY-MM-DD weekday hh:mm:ss AM/PM][昵称（QQ号）]说:<内容>
```

聊群引用消息：
```plaintext
聊群普通消息格式+ （（（引用的消息）））：其他消息段解析后内容
```
聊群合并转发消息：
```plaintext
聊群普通消息格式+合并转发消息格式
```

<hr />

戳一戳消息：
```plaintext
\(戳一戳消息\) 昵称(QQ:qq号) 戳了戳你
```
</details>

### 对于源码

~~警告！本插件源码可能含有以下内容~~
 
<details><summary>

~~点我开始赤史~~
</summary>

《三角形具有稳定性》
```python
if:
    if:
        if:
            if:
            else:
        else:
    else:
else:
```
《无意义分支》
```python
if a:do()
else:pass

```
《如判断》
~~其实是懒得改陈年史山缩进，但为了不必要的麻烦，就这样了（~~
```python
if True:
    todo()
#或者说
while True:
    todo()
    break
```
</details>

## 特性（不是bug那个）

- OpenAI API 支持
- QQ群组聊天支持
- QQ私聊支持
- 群组AT触发
- API 开放
- 戳一戳消息触发支持
- 多模型切换选择
- （未来将会支持）黑名单/白名单
- 不同群内自定义聊天开关
- 不同聊群可设置的自定义补充Prompt
- 向超控（特定聊群）推送插件的错误日志
- 自定义Bot消息被撤回时缓解尴尬的推送
- 合并转发/引用消息解析支持（只会解析纯文本/at消息段，如果您不理解这是什么意思，你可以理解为丢给LLM的信息只有"@a 你好"这样的格式）

## 安装方式
0. **前提条件**
    - 确保安装nonebot-plugin-localstore
    ```bash
        pip install nonebot-plugin-localstore
    ```
    并在你的`pyproject.toml`中添加如下内容：
    ```toml
    plugins=["nonebot_plugin_localstore"]
    #添加"nonebot_plugin_localstore"
    ```
    或直接使用nb-cli:
    ```bash
    nb plugin install nonebot-plugin-localstore
    ```
1. **通过pip安装**
   - 确保已安装Python（版本>=3.9）。
    打开命令行工具，执行以下命令来安装插件：
     ```bash
     pip install nonebot-plugin-suggarchat
     ```
     
     

2. **通过PDM安装**
    ```bash
    pdm add nonebot-plugin-suggarchat
    ```
    
以上方法需要在你的`pyproject.toml`中的**plugins**`列表`添加如下内容：
```toml
plugins=["nonebot_plugin_suggarchat"]
#添加"nonebot_plugin_suggarchat"
```

3. **通过nb-cli安装**
    ```bash
    nb plugin install nonebot-plugin-suggarchat
    # 此方法不需要再单独安装localstore
    ```
## 配置文件

- **配置文件路径**：通常位于项目的运行目录的config目录下，文件名为`config.json`。
### **配置项说明**
<details><summary>点此展开</summary>

| 配置项                         | 类型                | 默认值        | 解释                                                         |  
|------------------------------|-------------------|--------------|------------------------------------------------------------|  
| `memory_length_limit`          | int               | 50           | 单会话允许存储的最大消息数（**如果您不知道这是什么意思，请不要修改**）                                       |  
| `enable`                       | bool               | **false**         | 是否启用聊天机器人（即该插件）                                          |  
| `poke_reply`                   | bool               | true         | 是否启用戳一戳回复功能                                          |  
| `private_train`                | dict               | { "role": "system", "content": "`<请修改这里的内容>`" } | 私聊的系统提示词                                     |  
| `group_train`                  | dict               | { "role": "system", "content": "`<请修改这里的内容>`" } | 群聊训的系统提示词                                     |  
| `enable_group_chat`            | bool               | true         | 是否启用群聊功能                                            |  
| `enable_private_chat`          | bool               | true         | 是否启用私聊功能                                            |  
| `allow_custom_prompt`          | bool               | true         | 是否允许自定义提示                                          |  
| `allow_send_to_admin`          | bool               | true         | 是否允许向管理员发送消息                                    |  
| `admin_group`                  | int               | 0            | 管理员群组的ID                                             |  
| `admins`                       | list[int]               | []           | 管理员用户的列表                                            |  
| `open_ai_base_url`             | string             | ""           | OpenAI协议 API URL                                        |  
| `open_ai_api_key`              | string             | ""           | OpenAI协议 API 密钥                                            |  
| `say_after_self_msg_be_deleted` | bool               | true         | 自己的消息被删除后是否回复                                  |  
| `group_added_msg`              | string             | "你好，我是Suggar，欢迎使用Suggar的AI聊天机器人，你可以向我提问任何问题，我会尽力回答你的问题，如果你需要帮助，你可以向我发送“帮助”" | 加入群组时发送的欢迎消息                                     |  
| `send_msg_after_be_invited`    | bool               | true         | 被邀请进群后是否发送消息                                        |  
| `after_deleted_say_what`       | list[str]               | [ "Suggar说错什么话了吗～下次我会注意的呢～", "抱歉啦，不小心说错啦～", ... ] | 消息被删除后随机回复的内容                                   |
| `use_base_prompt`       | bool               | true | 是否使用基本提示词（即让LLM理解消息段解析）                                   |
| `preset`       | string               | __main__ | 是否使用预设（在工作目录的models文件夹下，预设json格式参考下文（你的预设名**不能**设为`__main__`）午）                                   |
| `max_tokens`       | int               | 100 | 在单次时，LLM最多可以回复多少个token（即字数，如果你的模型提供商支持                                   |
| `model`       | str               | auto | 使用什么模型（具体看你的API提供商                                   |
| `parse_segments` | bool               | true | 是否解析消息段，此权重覆盖`use_base_prompt`（即at/合并转发等）                                   |

</details>

## 预设
预设文件位于工作目录的models文件夹下，预设文件为json格式，具体格式如下：

```json
    {
    "model":"auto",
    "name":"",
    "base_url":"",
    "api_key":""
    }
```
### 解释：
- `model`: 使用的模型，默认为auto，即自动选择。
- `name`: 预设的名字，用于在插件中选择使用。
- `base_url`: OpenAI协议 API URL，默认为空。
- `api_key`: OpenAI协议 API 密钥，默认为空。
## 预设使用方法
1. 在工作目录的models文件夹下，创建一个json文件，文件名必须与预设的名字一致。
2. 在json文件中，填写预设的内容。
3. 在插件配置文件中，将`preset`的值设置为预设的名字（我们更推荐你使用**指令**来切换而不是直接修改配置文件）。
## 指令使用方法
| 指令            |                    参数     |           解释     |
|--------------|-----------------|------------------------------|
| **/聊天菜单** 或 **/chat_menu**: | 无 | 显示聊天菜单| 
| **/del_memory**, **/失忆**, **/删除记忆**, **/删除历史消息**, **/删除回忆**| 无 | 删除聊天记忆 |
| **/enable_chat** 或 **/启用聊天** | 无  | 启用聊天功能 |
 | **/disable_chat** 或 **/禁用聊天**| 无 |禁用聊天功能
 | **/prompt** | `--(show)` 展示当前提示词，`--(clear)` 清空当前prompt，`--(set) <文字>`则设置提示词 | 设置或查看当前自定义提示词（适用于用户自定义补充提示词）
 | **/presets** | 无 | 查看可用的预设列表
 | **/set_preset** 或 **/设置预设** 或 **/设置模型预设** | `<预设名>`  |设置当前使用的预设

<details><summary>隐藏指令</summary>

## 隐藏指令

为什么会有？因为开发者为了DEBUG，会保留一些在开发过程中测试的指令，如果您只是普通用户，请忽略，也不推荐您去使用这些指令
~~主要是写的垃圾~~
。
| 指令            |                    参数     |           解释     |
|--------------|-----------------|------------------------------|
| **/debug** | 无 | 显示调试信息并推送 |


</datials>

## 讨论

QQ 交流群: [链接](https://qm.qq.com/q/PFcfb4296m)