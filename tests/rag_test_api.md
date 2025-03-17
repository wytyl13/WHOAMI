
## Request url: POST  # 查询当前用户（单个/所有）会话历史消息
http://1.71.15.121:8888/get_conversation_history

Parameters:
user_id: <str, not null>
conversation_id: <str>

### 查询当前用户单个会话 - 有历史消息
```
Request Body
{
    "user_id": "jfoefjojdfo",
    "conversation_id": "1111111111111111111111111"
}

Response status: 200   # 有历史消息
{
	"code": 200,
	"data": [
		{
			"conversation": "1111111111111111111111111",
			"messages": [
				{
					"role": "user",
					"content": "我是谁"
				},
				{
					"role": "assistant",
					"content": "，不过在与我对话时你完全可以称呼自己为“用户”。我是一个来自阿里云的大规模语言模型，名叫Qwen。我的主要功能是生成各种文本内容，比如文章、故事、诗歌等，并能够回答问题、创作音乐歌词和提供解决问题的思路等。如果你有任何想要交流的内容或问题，都可以随时告诉我哦！"
				},
				{
					"role": "user",
					"content": "我是谁"
				},
				{
					"role": "assistant",
					"content": "，但为了保护用户隐私，在与我的对话中你选择不透露个人信息。我是一个来自中国的大语言模型，名叫Qwen，由阿里云开发，很高兴能和你交流。你可以问我任何问题，我会尽力帮助你。"
				}			]
		}
	],
	"is_error": false,
	"is_success": true,
	"extra": null,
	"time_stamp": 1741749375.8817337
}
```


### 查询当前用户单个会话 - 无历史消息
```
Request Body
{
    "user_id": "jfoefjojdfo",
    "conversation_id": "11"
}
Response status: 200 # 单个查询无历史消息
{
	"code": 200,
	"data": [],
	"is_error": false,
	"is_success": true,
	"extra": null,
	"time_stamp": 1741749582.968292
}
```


### 查询当前用户单个会话 - 传参错误
Request Body
{
    "user_id": "",
    "conversation_id": "11"
}

Response status: 1 # 请求错误
{
	"code": 1,
	"data": "user_id must not be null!",
	"is_error": true,
	"is_success": false,
	"extra": null,
	"time_stamp": 1741749706.9948413
}


### 查询当前用户所有会话 - 有历史消息
```
Request Body
{
    "user_id": "jfoefjojdfo"
}

Response status # 200
{
	"code": 200,
	"data": [
		{
			"conversation_id": "1111111111111111111111111",
			"messages": [
				{
					"role": "user",
					"content": "我是谁"
				},
				{
					"role": "assistant",
					"content": "，不过在与我对话时你完全可以称呼自己为“用户”。我是一个来自阿里云的大规模语言模型，名叫Qwen。我的主要功能是生成各种文本内容，比如文章、故事、诗歌等，并能够回答问题、创作音乐歌词和提供解决问题的思路等。如果你有任何想要交流的内容或问题，都可以随时告诉我哦！"
				},
				{
					"role": "user",
					"content": "我是谁"
				},
				{
					"role": "assistant",
					"content": "，但为了保护用户隐私，在与我的对话中你选择不透露个人信息。我是一个来自中国的大语言模型，名叫Qwen，由阿里云开发，很高兴能和你交流。你可以问我任何问题，我会尽力帮助你。"
				}
			]
		},
		{
			"conversation_id": "111",
			"messages": [
				{
					"role": "user",
					"content": "我来自哪里"
				},
				{
					"role": "assistant",
					"content": "您可以在对话中直接告诉我您来自哪里，比如城市、国家等。如果您不想透露具体位置，也可以告诉我您对哪个地方感兴趣，我可以帮您提供相关信息。"
				}
			]
		}
	],
	"is_error": false,
	"is_success": true,
	"extra": null,
	"time_stamp": 1741751408.4379342
}
```


### 查询当前用户所有会话 - 无历史消息
```
{
    "user_id": "jfoefjojdfo1"
}

Response status # 200
{
	"code": 200,
	"data": [],
	"is_error": false,
	"is_success": true,
	"extra": null,
	"time_stamp": 1741751492.7837582
}
```




## Request url: POST  # Chat请求
http://1.71.15.121:8888/chat_health_report
Parameters:
question: <str, not null>
user_id: <str, not null>
conversation_id: <str, not null>
messages: <list>
### 请求成功
```

{
    "question": "我来自哪里",
    "user_id": "jfoefjojdfo",
    "conversation_id": "111",
    "messages": [

        {
            "role": "user", 
            "content": "我是谁"
        },
        {
            "role": "assistant", 
            "content": "您是提问的用户。在我们的对话中，您可以告诉我您的名字或者提供更多背景信息，这样我就能更好地为您提供帮助或进行交流了。"
        }
    ]
}


Response status # 流式返回，需要对接
data: 您

data: 可以通过

data: 提供

data: 一些

data: 具体

data: 的信息

data: 来

data: 告诉我

data: 您

data: 来自

data: 哪里

data: ，

data: 比如

data: 城市

data: 、

data: 国家

data: 或者其他

data: 相关信息

data: 。

data: 不过

data: 根据

data: 当前

data: 的

data: 对话

data: 内容

data: ，

data: 我没有

data: 足够的

data: 信息

data: 来

data: 确定

data: 您的

data: 地理位置

data: 。

data: 如果您

data: 愿意

data: 分享

data: ，

data: 我很

data: 乐意

data: 听到

data: 关于

data: 您的

data: 更多信息

data: ！

```


### 请求失败
{
    "question": "",
    "user_id": "jfoefjojdfo",
    "conversation_id": "111",
    "messages": [

        {
            "role": "user", 
            "content": "我是谁"
        },
        {
            "role": "assistant", 
            "content": "您是提问的用户。在我们的对话中，您可以告诉我您的名字或者提供更多背景信息，这样我就能更好地为您提供帮助或进行交流了。"
        }
    ]
}

Response status # 1
{
	"code": 1,
	"data": "question must not be null",
	"is_error": true,
	"is_success": false,
	"extra": null,
	"time_stamp": 1741751976.6568792
}

# 请求前先访问当前会话所有历史消息，生成请求参数。