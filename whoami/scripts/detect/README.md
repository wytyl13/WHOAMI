# 视频流目标检测
```
深度学习模型：yolov10
topic_list = ["/fire/smoke/warning", "/fallen/falling/warning"]

硬件占用：每路视频流执行单个任务占用显存600M

任务1：烟火预警
topic: /fire/smoke/warning
预测类别: 火 烟雾
训练数据个数：2205
训练批次: 4
epoch: 185
prediction:  0.79
recall: 0.66

任务2：跌倒预警
topic: /fallen/falling/warning
预测类别: 站立、跌倒、正在跌倒
训练数据个数: 5366
训练批次: 4
epoch: 250
prediction: 0.97
recall: 0.95
```


# 接口文档
## 可订阅主题查看
在线地址：https://doc.apipost.net/docs/detail/396d7beef806000?target_id=16d78d69346153&locale=zh-cn
**接口URL**

> http://192.168.12.18:8888/list_all_topic

| 环境  | URL |
| --- | --- |


**请求方式**

> GET

**Content-Type**

> none

**认证方式**

> 继承父级

**响应示例**

* 成功(200)

```javascript
{
	"code": 200,
	"data": [
		"/fallen/falling/warning"
	],
	"is_error": false,
	"is_success": true,
	"extra": null,
	"time_stamp": 29
}
```

| 参数名 | 示例值 | 参数类型 | 参数描述 |
| --- | --- | ---- | ---- |
| code | 200 | Number | 状态码 |
| data | - | Array | 数据内容 |
| is_error | false | Boolean | 是否发生错误 |
| is_success | true | Boolean | 是否成功 |
| extra | - | Null | 额外信息 |
| time_stamp | 29 | Number | 时间戳 |

* 失败(404)

```javascript
暂无数据
```




## 视频流解析接口文档
在线地址：https://doc.apipost.net/docs/detail/39097d2900e4000?target_id=f191a29f4605f&locale=zh-cn
**接口URL**

> http://192.168.12.18:8888/fire_smoke_warning

| 环境  | URL |
| --- | --- |


**请求方式**

> POST

**Content-Type**

> json

**请求Body参数**

```javascript
{
	"device_sn": "BD0632136",
	"topic_list": [
		"/fallen/falling/warning", "/fire/smoke/warning"
	],
	"sampling_interval": 85
}
```

| 参数名 | 示例值 | 参数类型 | 是否必填 | 参数描述 |
| --- | --- | ---- | ---- | ---- |
| device_sn | BD0632136 | String | 是 | 设备序列号 |
| topic_list | - | Array | 否 | 主题列表.["/fallen/falling/warning", "/fire/smoke/warning"] |
| topic_list.0 | /fallen/falling/warning | String | 是 | topic1 |
| topic_list.1 | /fire/smoke/warning | String | 是 | topic2 |
| sampling_interval | 85 | Number | 否 | 采样间隔，默认为1秒 |

**认证方式**

> 继承父级

**响应示例**

* 成功(200)

```javascript
{
	"code": 200,
	"data": "****视频流解析成功****开始后台执行！****",
	"is_error": false,
	"is_success": true,
	"extra": null,
	"time_stamp": 2
}
```

| 参数名 | 示例值 | 参数类型 | 参数描述 |
| --- | --- | ---- | ---- |
| code | 200 | Number | 正确状态码 |
| data | 视频流解析成功****视频流解析成功****开始后台执行！**** | String | 数据内容 |
| is_error | false | Boolean | 是否发生错误 |
| is_success | true | Boolean | 是否成功 |
| extra | - | Null | 额外信息 |
| time_stamp | 2 | Number | 时间戳 |

* 失败(1)

```javascript
{
	"code": 1,
	"data": "视频流正在后台解析！****",
	"is_error": true,
	"is_success": false,
	"extra": null,
	"time_stamp": 1732780463.2507458
}
```

| 参数名 | 示例值 | 参数类型 | 参数描述 |
| --- | --- | ---- | ---- |
| code | 1 | Number | 正确状态码 |
| data | 视频流正在后台解析！******* | String | 错误内容 |
| is_error | true | Boolean | 是否发生错误 |
| is_success | false | Boolean | 是否成功 |
| extra | - | Null | 额外信息 |
| time_stamp | 1732780463.2507458 | Number | 时间戳 |






## 单个视频流解析状态查看接口文档
在线地址：https://doc.apipost.net/docs/detail/390999aa1806000?target_id=10529d7e346093&locale=zh-cn
**接口URL**

> http://192.168.12.18:8888/check_video_stream_url_status

| 环境  | URL |
| --- | --- |


**请求方式**

> POST

**Content-Type**

> json

**请求Body参数**

```javascript
{
	"video_stream_url": "******"
}
```

| 参数名 | 示例值 | 参数类型 | 是否必填 | 参数描述 |
| --- | --- | ---- | ---- | ---- |
| video_stream_url | **** | String | 是 | 视频流地址 |

**认证方式**

> 继承父级

**响应示例**

* 成功(200)

```javascript
{
	"code": 200,
	"data": [
		"[ \"/fallen/falling/warning\", \"/fire/smoke/warning\"]"
	],
	"is_error": false,
	"is_success": true,
	"extra": null,
	"time_stamp": 121212
}
```

| 参数名 | 示例值 | 参数类型 | 参数描述 |
| --- | --- | ---- | ---- |
| code | 200 | Number | 成功状态码 |
| data | - | Array | 数据内容 |
| data.0 | /fallen/falling/warning | String | - |
| data.1 | /fire/smoke/warning | String | - |
| is_error | false | Boolean | 是否发生错误 |
| is_success | true | Boolean | 是否成功 |
| extra | - | Null | 额外信息 |
| time_stamp | 121212 | Number | 时间戳 |

* 失败(404)

```javascript
暂无数据
```


## 获取所有视频流解析状态接口文档
在线地址：https://doc.apipost.net/docs/detail/396d70d82c06000?target_id=12ef6548f46067&locale=zh-cn
**接口URL**

> http://192.168.12.18:8888/list_all_process

| 环境  | URL |
| --- | --- |


**请求方式**

> GET

**Content-Type**

> none

**认证方式**

> 继承父级

**响应示例**

* 成功(200)

```javascript
{
	"code": 200,
	"data": {
		"****": [
			"/fire/smoke/warning"
		]
	},
	"is_error": false,
	"is_success": true,
	"extra": null,
	"time_stamp": 82
}
```

| 参数名 | 示例值 | 参数类型 | 参数描述 |
| --- | --- | ---- | ---- |
| code | 200 | Number | 状态码 |
| data | - | Object | 数据内容 |
| data.****** | - | Array | 视频流地址 |
| is_error | false | Boolean | 是否发生错误 |
| is_success | true | Boolean | 是否成功 |
| extra | - | Null | 额外信息 |
| time_stamp | 82 | Number | 时间戳 |

* 失败(404)

```javascript
暂无数据
```

