# import cv2
# import time
# import requests

# cap = cv2.VideoCapture("https://open.ys7.com/v3/openlive/BD3202818_1_2.m3u8?expire=1740641527&id=814882500458319872&t=455106fe74399439794cf3fe4ba7e25ab9b0c55feb5b8216502d26236d208154&ev=100")
# cap.set(cv2.CAP_PROP_FPS, 30)


# device_sn = [
#     "BD3202818",
#     "BD0632093",
#     "BE6992578",
#     "BC8197657",
#     "BE3337110",
# ]




if __name__ == "__main__":
    #!/usr/bin/env python3
    # -*- coding: utf-8 -*-
    import pandas as pd
    import pymysql
    import sys
    from datetime import datetime

    def export_to_excel():
        # 数据库连接配置
        config = {
            'host': '192.168.0.10',
            'port': 3366,
            'user': 'root',
            'password': '2xryuf@I73T',
            'database': 'shunxikeji',
            'charset': 'utf8mb4'
        }
        
        try:
            print("正在连接数据库...")
            connection = pymysql.connect(**config)
            
            print("正在查询数据...")
            query = "SELECT * FROM sx_institution_elderly_bed"
            df = pd.read_sql(query, connection)
            connection.close()
            
            print(f"查询到 {len(df)} 条记录")
            
            # 导出为XLSX
            today = datetime.now().strftime('%Y%m%d')
            filename = f'/work/ai/WHOAMI/elderly_bed_{today}.xlsx'
            
            print("正在导出Excel文件...")
            df.to_excel(filename, index=False, engine='openpyxl')
            
            print(f'✅ 数据已成功导出到: {filename}')
            
            # 显示前几行数据预览
            print("\n数据预览:")
            print(df.head())
            
        except Exception as e:
            print(f'❌ 导出失败: {e}')
            sys.exit(1)

        export_to_excel()
    # https://api.map.baidu.com/weather/v1/?district_id=222405&data_type=all&ak=lEnj3LGZkkyUYhdkIF0yYcfw603jq284
    
    
    
    # while cap.isOpened():
    #     ret, frame = cap.read()
    #     current_time = time.time()
    #     local_time = time.localtime(current_time)
    #     formatted_time = time.strftime('%Y-%m-%d %H:%M:%S', local_time)
    #     if not ret or frame is None or frame.size == 0:
    #         raise ValueError("frame is invalid!")
    #     else:
    #         print(f"------------------------------------{formatted_time}")
    
    # from diagrams import Diagram, Edge
    # from diagrams.custom import Custom

    # with Diagram("老龄化-心理资本-职业承诺模型", show=False):
    #     aging = Custom("老龄化压力", "./aging_icon.png")
    #     capital = Custom("积极心理资本", "./psych_icon.png")
    #     commit = Custom("职业承诺", "./commit_icon.png")
        
    #     aging >> Edge(label="β=-0.42**", color="red") >> capital
    #     capital >> Edge(label="β=0.67***", color="darkgreen") >> commit
    
    # from modelscope.pipelines import pipeline
    # from modelscope.utils.constant import Tasks
    # ans = pipeline(
    #     Tasks.acoustic_noise_suppression,
    #     model='iic/speech_frcrn_ans_cirm_16k'
    # )
    # result = ans(
    #     '/work/ai/WHOAMI/whoami/tool/tts/output/1a42f4b3-94a4-4cee-a5e5-16bf820a29d3.wav',
    #     output_path='output.wav')