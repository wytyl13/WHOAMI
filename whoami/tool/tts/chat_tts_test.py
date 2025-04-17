import ChatTTS
import torch
import torchaudio

chat = ChatTTS.Chat()
chat.load(compile=False, custom_path='/root/.cache/modelscope/hub/models/AI-ModelScope/ChatTTS') # Set to True for better performance

rand_spk = chat.sample_random_speaker()
print(rand_spk) # save it for later timbre recovery
rand_spk = "蘁淰敝欀旄伪紓彄忀荂蘁爻諧亚哵蝋彟賤葱赑萖膺縷蓔豉埼籑莔琇妲呧匐俨狵懱狿瑏绑蛣喆弣哓杝蟤槸禺攈扝坩橃氷唩疪暖嚥凁朴菜淺垰緂夈粙蔷水猿裼旹蓘澘澥勱訕巕搢蓗久崰潔帥嘥絷剏惋乀趴跡箐止毑絁粶莑桝趓冣礃獥瀐湭磯乁娲糙徾寜栏肄蕴蔮妜庣挆瀭譊救撯仞曍痔硴瞫習樗瑈玐憶蔣珑蠎纗笆苶媼斀揘孏湟憞蓺滊湚佤剟凣楃泈尰慅咜怘嶻瘬孶甒兛尓荔棻暲箝纁礐譨袠殖湒珌笪蛉誡峓旷层族憟殤跗脋灯捶荈義吟臾緄蝮筅禛析蟺撳賦烞竓沋厩栦烠喓薜憅冖徔斉剩灲北炽絁媺乙覃扼糊蠃畁蛙请埭禲囶螉甃赺惤级谱剌捐皫襆孇供訤瓆猈箈卞暾曻垳執礘搫琰巖彨莬茆裓浡嶅瑌悊喌緞洱嬀粒拤谵癧脓苧正膣褥夭愪聀撵渜脌窯慩蓹貟觨聐賐堃舢仁决崶跌檔粶敍瀈政拋澠敔蒅猲儵冚攓層硣喒爁傹厱聥瀶怺浪歖噦湒壢戹疐图囌稸蚧塝痖耀初烸誎憷抃暈舗肹螹整缞挥例祭紴勽丙祳碃募泖苀卼賯索粣赾谶坟溏吝蘳讃槚惬肨袬毀贂笃買斪涾棩稞簝箐沑羳痒棷缆蠥晆蟝蔓紭介俺燉汞娸熤煳欳筍渊懒椷獛宀岥玺罸苄墱绊擢嵻縲朽僲匦瘮街硽奀劂糦荙筈婡忧栆詹虂蔋縉萬埠洱犖朢弋絪徫烣戊媎偊皝炭甖侄炑槸埣汏王伜硾惵螻礽枒稼妒璓巢朜堔広腯蝍廐覶搲妩装瘼竺篛粑搳紈璖奣傅赯觤俊乧缟薡嶮袄宵瘘涋票拭帜袎哕勩孲伨祷纠淄佒譜烔曝剈様厛浓蕅艑圡聞搬艹荘唑胤汧儣璭浖宊垅姚垻歡睯礦症塚倢覇慍圯榍荹樼跴昔纵死兗唞瓑裺慾厨棤哣榭僇窾粇膓瀪剈嬻蟘嗜偨咜嗴讣街盒盀蔻覉灭烾嘖珧舐漥蘞毷祢圇孶办虵穂奛着娟蛚梹昳榶劰峆耞琒崄虪褣貐詆徸皀歧茽咝瀌硴毧赱瓒婿睢呡张搰暟矱僋弦柬嫀瓽擧衈徏箣獃湸蒎狾簛寜譬聡茵嬎彁瀒艕殈授倦曟舂妠嬚幡絭到茼緕耓璌桞丆覲嫸姸宭搗斬犰脌萣墀苎斀巩紵咙坜噀桚渨耱棦倭娏竬虾桡漇蕯梨訦膿炨烺纃楛蛇擰縑恌喞綫湌熜漙崩奧扆愝殴菐悭腎熷簎柙実拰聍榧庚螣旚疿咭祐帗望旯繲檗猎柷諔紈褌艊嚥皵腮漛肯芴誺磚疡州佷撞緯泓暁姭偞蕌毃茖盖磜糌籠怮祯歘琕垩煎虯畍撽讣孿祸容乊坪秫狘帔聙璲媸腸圍虻膌灴璚咺掜琿啩檂巐贔崴條罵橁囟娈覺堁矖蘻莥梖昇幉蔕廁趽煉繳坩琣蒄甼冲欘籛堓暂侑勯偌貆燃瘎繥皹密匭氻磌党嶊溹灓梹纛荟絴覄絅匠捕侉瓟狦猣痈衁溣獓磙盖宍椼罤剥瀞摴漑牛肸呇噟嗳搷緃胏炀一㴅"
params_infer_code = ChatTTS.Chat.InferCodeParams(
    spk_emb = rand_spk, # add sampled speaker 
    temperature = .3,   # using custom temperature
    top_P = 0.7,        # top P decode
    top_K = 20,         # top K decode
)

params_refine_text = ChatTTS.Chat.RefineTextParams(
    prompt='[oral_2][laugh_0][break_6]',
)

texts = [
    """我们公司位于山西省运城市盐湖区黄河金三角科创城C四楼三F 设有总部售后服务中心 如果您想参观或有其他相关需求 请提前联系我们进行预约。""", 
    "您询问的是前天晚上即二零二五年四月七日的睡眠情况 根据提供的数据。",
    "以下是您在即二零二五年四月七日的睡眠分析 上床时间 十九点 入睡时间 十九点 醒来时间 次日凌晨两点。",
    "总监测时长约十一小时五十七分钟。从这些数据可以看出 您在二零二五年四月七日的睡眠质量不是很好。建议您可以尝试调整作息时间。",
    "改善睡眠环境或咨询医生以获得更好的睡眠质量。", 
    "PUT YOUR 2nd TEXT HERE"
]

wavs = chat.infer(
    texts,
    params_refine_text=params_refine_text,
    params_infer_code=params_infer_code,
)

for i in range(len(wavs)):
    """
    In some versions of torchaudio, the first line works but in other versions, so does the second line.
    """
    try:
        torchaudio.save(f"basic_output{i}.wav", torch.from_numpy(wavs[i]).unsqueeze(0), 24000)
    except:
        torchaudio.save(f"basic_output{i}.wav", torch.from_numpy(wavs[i]), 24000)