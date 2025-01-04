# health report fast api

## request

```
request url: http://192.168.12.18:8000/sleep_indices (局域网)

request json:
{
	"device_sn": ["13D7F349200080712111150807", "13D1F349200080712111151107"], # 设备编号（字符串或list）
	"query_date": "2024-12-25" # 日期
}

response json:
{
	"code": 200,
	"data": "To start process background. It will take approximately 3.0 minutes.",
	"is_error": false,
	"is_success": true,
	"extra": null,
	"time_stamp": 1735290362.7962584
}

后台自动处理并将处理结果写入数据库

```

## result details

```
# 绘图数据起始时间戳区间固定
{
    'total_num_second_on_bed': 38342, # 总在床时长（秒）
    'sleep_second': 28559,  # 睡眠时长（秒）
    'deep_sleep_second': 6734, # 深度睡眠时长（秒）
    'waking_second': 8802,  # 清醒时长（秒）
    'to_sleep_second': 1956.6, # 入睡时长（秒）
    'total_num_hour_on_bed': '10小时39分钟', # 总在床时长（小时）
    'sleep_hour': '7小时55分钟',  # 睡眠时长（小时）
    'deep_sleep_hour': '1小时52分钟',  # 深度睡眠时长（小时）
    'waking_hour': '2小时26分钟',  # 清醒时长（小时）
    'to_sleep_hour': '0小时32分钟', # 入睡时长（小时）
    'waking_count': 5,  # 夜醒次数
    'on_bed_time': '2024-12-24 20:54:35',  # 上床时间（节点）
    'sleep_time': '2024-12-24 21:10:55',  # 入睡时间（节点）
    'waking_time': '2024-12-25 08:28:51',  # 醒来时间（节点）
    'sleep_stage_image_x_y': '[[[1735044875, 1735045854], [1735045855, 1735047327], [1735047328, 1735053674], [1735053675, 1735055272], [1735055273, 1735055917], [1735055918, 1735058970], [1735058971, 1735060065], [1735060066, 1735060977], [1735060978, 1735062574], [1735062604, 1735064702], [1735064703, 1735066591], [1735066592, 1735070342], [1735070343, 1735071322], [1735071323, 1735074111], [1735074112, 1735075278], [1735075279, 1735077295], [1735077296, 1735077899], [1735077900, 1735080461], [1735080462, 1735081051], [1735081052, 1735083337], [1735083338, 1735084775], [1735084776, 1735085585], [1735085586, 1735086531], [1735086532, 1735088399]], [3, 2, 3, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 3, 2, 1, 2, 3, 2, 3, 4]]',  # 睡眠分区绘图
    'sleep_efficiency': 0.74, # 睡眠效率
    'deep_sleep_efficiency': 0.24, # 深睡效率
    'leave_count': 0,  # 离床次数
    'leave_bed_time': '2024-12-25 08:59:59',  # 离床时间
    'total_num_second': 43581,  # 总监测时长（秒）
    'total_num_hour': '12小时6分钟',  # 总监测时长（小时）
    'query_date': '2024-12-25',  # 查询日期
    'save_file_path': 'none',  # 绘图url
    'device_sn': '13D7F349200080712111150807', # 设备编号
    'average_breath_bpm': 14.07,  # 平均呼吸率
    'max_breath_bpm': 23.01,  # 最大呼吸率
    'min_breath_bpm': 8.44,  # 最小呼吸率
    'average_heart_bpm': 87.67,  # 平均心率
    'max_heart_bpm': 121.88,  # 最大心率
    'min_heart_bpm': 6.0,  # 最小心率
    'body_move_count': 336,  # 体动次数
    'body_move_exponent': 11.2,  # 体动指数
    'body_move_image_x_y': '[[1735044604, 1735046645, 1735048443, 1735050241, 1735052039, 1735053837, 1735055635, 1735057433, 1735059231, 1735061029, 1735062827, 1735064625, 1735066423, 1735068221, 1735070019, 1735071817, 1735073615, 1735075413, 1735077211, 1735079009, 1735080807, 1735082605, 1735084403, 1735086201, 1735087999, 1735088399], [0, 22, 18, 25, 26, 26, 9, 4, 15, 10, 22, 10, 16, 5, 6, 0, 16, 4, 6, 20, 12, 5, 23, 34, 2, 0]]', # 体动绘图
    'breath_exception_count': 0,  # 呼吸异常次数
    'breath_exception_image_x_y': '[[1735044604, 1735088399], [0, 0]]',  # 呼吸异常绘图
    'score': 56.26,  # 评分
    'score_name': '较差', # 评分归类
    "consist_count_waking": 1, # 连续?晚夜醒时长超过31分钟
    "consist_count_sleep_efficiency": 1, # 连续?晚睡眠效率小于80%
}
```

## 开发日志
```
# 对原始数据的理解
1 body_move_data != 0 体动  体动情况下有心率呼吸率数据，因此也要对体动进行睡眠分区
2 distance 和 signal_intensity 成正比 但是不绝对（存在有距离但是信号强度为0的情况）
3 信号强度signal_intensity（未使用）信号强度为0时离床
    信号强度不为0一定有数据，距离不为0不一定有数据，信号为0也可能有数据（但是少数13D2F34920008071211195ED07）
    signal_intensity == 0 and state == 0 不存在
    signal_intensity == 0 and breath_bpm != 0 不存在
    signal_intensity == 0 and inout_bed == 1 存在少数（13D2F34920008071211195ED07）
    signal_intensity != 0 and inout_bed == 0 存在少数（13D4F349200080712111955907）
    signal_intensity != 0 and breath_bpm == 0 存在少数但是属于憋气状态（因此根据信号强度来判断在离床最准确）
    signal_intensity == 0 and inout_bed is none 存在多数
    signal_intensity == 0 and in_out_bed is none and breath_bpm != 0 存在少数（可以忽略）
4 in_out_bed 在离床
    in_out_bed == 1 and heart_bpm == 0 不存在
    in_out_bed == 0 and heart_bpm != 0 不存在
    in_out_bed is none and heart_bpm != 0 存在多数
5 决定：在离床判断条件 
    signal_intensity == 0 and inout_bed != 1  离床
    signal_intensity !=0 or (signal_intensity == 0 and inout_bed == 1) 在床
6 state稳定性（未使用，通过分析原始数据发现，该稳定性数值本身并不稳定）
    signal_intensity=0 and state=0 数据不稳定
    刚开始使用其判断体动，后引入体动动量值指标，所以抛弃
7 睡眠分区要基于在床数据进行分析
    1 首先对原始数据使用4条件分类为离床和在床

    2 基于在床数据做睡眠分区并统计睡眠数据
        使用统计方式分区

    3 然后基于在床数据去进行体动统计
    signal_intensity is none and body_move_data is not none 体动
    查询对应的device_sn和create_time然后将对应的体动数据赋值给对应的条目
    统计一个总体动次数
    以20分钟为一个批次统计体动次数并报告统计数据

    4 判断呼吸异常事件
    呼吸异常 "breath_bpm": [10, 22]  标准区间，该标准区间由后端使用数据库维护
    不在区间范围内定义为呼吸异常，报告该呼吸异常60秒内的呼吸率数据

8 
```
v1.0 
完成：在离床  

已完成：
1、各项基本指标的获得
2、基于统计方法的睡眠分区
3、简单的睡眠建议

下一步：
1、使用体动能量值这个数据区进行体动数据的分析
2、优化统计方法的睡眠分区算法，然后得到真正的标注数据
3、使用标注数据和时间序列深度学习模型训练预训练睡眠分区模型
4、更为精细的睡眠建议和更深入的健康建议
```
