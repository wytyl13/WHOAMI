import sys
import os
import uuid
import shutil
import re
import torch
import torchaudio
import logging
import asyncio
from datetime import datetime
import concurrent.futures
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Request, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any, Generator, Tuple
from pydantic import BaseModel
import ChatTTS
import torch
import torchaudio
import cn2an
import numpy as np
import traceback


chat = ChatTTS.Chat()
chat.load(compile=False, custom_path='/root/.cache/modelscope/hub/models/AI-ModelScope/ChatTTS') # Set to True for better performance
# rand_spk = chat.sample_random_speaker()
# print(rand_spk) # save it for later timbre recovery
#1 "蘁淰敡欀囄幀壎编儂藮繪见漬薬涿蚲糥丽忋咐瀝囃偶孕櫯謓岢秱浱捠嬦厲嶉壥菰缩橒誂掫獈疻汔仄兩漲痝崚梺竑刨機熜召粏佾眢沈浉朮豱悝締廉嘂卮瑩瓕亴戁膸塉上傄瘢翼碠湣燛凳唜潏篋濺澣藶虂嘢粓聎脋拺泭渾擟脎哤勤攟灖垭曦宵缹渐臹廋耉珪浅贞讟葽游寪婹汑蔢嫯濯襌慠享畱嬄凇眲懤莺执硐嬫牉謵厪而渥撊幭猿妟嚍稰質萃肝包冖啱杣墱臣痖燢禭聀宸詤悋眭判盀讕篕丼埨襲庾堪妩定虗矦誯舵瑪緤臽引伬噻侳漳欹罤烤槮寷裫螵眷盋値凘僁朹慗薞筿寬止貼宧纁脆箅双丞搃瓼聇椅趔搁佻琐垫泍揽莓螆噛淈媽喒箏蚖篘琞獻涙唔忙俤冘琼誇庑垥佪劍嗷蜁謹贤吅実姣瀳秇莁硃葢蚳眾蝱材猑藹攵拁藪冷愙偕芘獔劝蕍燰埙官瀰渞瑔耏扗屌栛衴扵旍聈榱岘缵幙塋貚狄虉嶨栫菩叕胃檳樀癇弓璀舎赉抦倇璴螾萼屐滵孲溪擟廾則袜惡罕嵋渡棖蝬懜蔭硲嚧琣糔圄摗硝待炡葞赡劌豣灌肑皒蔪豧興肄坚蠑荝瓻費彶蜶簔脕歌艍持玹蛝津桲萿増渪牳琁貏嚁嵏愷嵯幹磨琒憞苌梩戬腆褱腃僛意紦縕蒹糠纋斈譚卆尒缋窆琡罅東赋嚡说枎俐层牛紁穪毻舑誩晷倈罩製奘貔蓺败焪旂笹戡崘艑嶭走悏罫萉嶲襍癵瓼蜣敷縍穙宐痄埭謦潝纩衬涡误挀歒瀸廋夌熦珀甪堐僘磮癄沽憾懇袅蟞狛碃擒復厬藌襔秱裱簜峮賡凮夢記瀭痏愤砞攳謍揣訡絙崑脔賰嚴桜攎觢薡玙蒰奣蒗皜殇纷勹歡猏璗偍嗼葩佚紂哳划窇諎俀荼笏泠欱抙帡枻胳梮趭昩刞孷縞缜塥蘃穬墙皭哬紤捠恨瓡搕您烺瑶摎汪槓崇蠋甏设獑蟛蔉嗢抦磦硰眻蘍廏叝疫佔縄臃夠窃峌东抷裛谩犵藴氐瓙凂蛌扯段弸苸吁舰居栻祇喷禟蛞澃榔敺癗园貸廬炽謰缿硟丫畾剱皟趽誉煉缰枏搻篡菎禪场烠潤橵趎澴笑炩咊寂蝍蛗疂濽掩澂珣徔唟垱剷崅從觷惧襱墥抷兵琁莳淩衾檝湁掣珞碩朻挖嫗梾茇癐决奸懋弿詈畍觩竾煀录矷宅泈莇卟箬帘夲拫蠈峒覓瘥芐繴媻罺珏懥篷艉栥谅唃聱凟携崛围巀粳痻砲痞乹夢秮趐匨莥場昨肸熝疅昿崖穡倒控猚稕禮夘胸棈噍捱傶怌绽褃瑋茼蚈幊卝諹伖娓窽泹欳昊瞲妐獀椕礒緺嶁藩毗猱肃緩臊萬諂悦矢揚灜蔢暐嶙刮愐湹篠琉諈賆褤挂疠搚婰獎簃璑謅久蟑愰蚌媑蓨粟你蚲嵻宔茉豄湻噏衷惊寥蔿圎艌悧啿橭臯卖湅斀卷沣毻污坠柈儂糌矑元摛愨谡怬讥炻窌刮悟菱舺晸縸傕凝粌唷蔗号叽穝硄綧粙歸兤瑥煡嫇螂湗卵砆误篿训畸癖坵嬊筮猌一㴆"


# rand_spk = "蘁淰敝欀旄伪紓彄忀荂蘁爻諧亚哵蝋彟賤葱赑萖膺縷蓔豉埼籑莔琇妲呧匐俨狵懱狿瑏绑蛣喆弣哓杝蟤槸禺攈扝坩橃氷唩疪暖嚥凁朴菜淺垰緂夈粙蔷水猿裼旹蓘澘澥勱訕巕搢蓗久崰潔帥嘥絷剏惋乀趴跡箐止毑絁粶莑桝趓冣礃獥瀐湭磯乁娲糙徾寜栏肄蕴蔮妜庣挆瀭譊救撯仞曍痔硴瞫習樗瑈玐憶蔣珑蠎纗笆苶媼斀揘孏湟憞蓺滊湚佤剟凣楃泈尰慅咜怘嶻瘬孶甒兛尓荔棻暲箝纁礐譨袠殖湒珌笪蛉誡峓旷层族憟殤跗脋灯捶荈義吟臾緄蝮筅禛析蟺撳賦烞竓沋厩栦烠喓薜憅冖徔斉剩灲北炽絁媺乙覃扼糊蠃畁蛙请埭禲囶螉甃赺惤级谱剌捐皫襆孇供訤瓆猈箈卞暾曻垳執礘搫琰巖彨莬茆裓浡嶅瑌悊喌緞洱嬀粒拤谵癧脓苧正膣褥夭愪聀撵渜脌窯慩蓹貟觨聐賐堃舢仁决崶跌檔粶敍瀈政拋澠敔蒅猲儵冚攓層硣喒爁傹厱聥瀶怺浪歖噦湒壢戹疐图囌稸蚧塝痖耀初烸誎憷抃暈舗肹螹整缞挥例祭紴勽丙祳碃募泖苀卼賯索粣赾谶坟溏吝蘳讃槚惬肨袬毀贂笃買斪涾棩稞簝箐沑羳痒棷缆蠥晆蟝蔓紭介俺燉汞娸熤煳欳筍渊懒椷獛宀岥玺罸苄墱绊擢嵻縲朽僲匦瘮街硽奀劂糦荙筈婡忧栆詹虂蔋縉萬埠洱犖朢弋絪徫烣戊媎偊皝炭甖侄炑槸埣汏王伜硾惵螻礽枒稼妒璓巢朜堔広腯蝍廐覶搲妩装瘼竺篛粑搳紈璖奣傅赯觤俊乧缟薡嶮袄宵瘘涋票拭帜袎哕勩孲伨祷纠淄佒譜烔曝剈様厛浓蕅艑圡聞搬艹荘唑胤汧儣璭浖宊垅姚垻歡睯礦症塚倢覇慍圯榍荹樼跴昔纵死兗唞瓑裺慾厨棤哣榭僇窾粇膓瀪剈嬻蟘嗜偨咜嗴讣街盒盀蔻覉灭烾嘖珧舐漥蘞毷祢圇孶办虵穂奛着娟蛚梹昳榶劰峆耞琒崄虪褣貐詆徸皀歧茽咝瀌硴毧赱瓒婿睢呡张搰暟矱僋弦柬嫀瓽擧衈徏箣獃湸蒎狾簛寜譬聡茵嬎彁瀒艕殈授倦曟舂妠嬚幡絭到茼緕耓璌桞丆覲嫸姸宭搗斬犰脌萣墀苎斀巩紵咙坜噀桚渨耱棦倭娏竬虾桡漇蕯梨訦膿炨烺纃楛蛇擰縑恌喞綫湌熜漙崩奧扆愝殴菐悭腎熷簎柙実拰聍榧庚螣旚疿咭祐帗望旯繲檗猎柷諔紈褌艊嚥皵腮漛肯芴誺磚疡州佷撞緯泓暁姭偞蕌毃茖盖磜糌籠怮祯歘琕垩煎虯畍撽讣孿祸容乊坪秫狘帔聙璲媸腸圍虻膌灴璚咺掜琿啩檂巐贔崴條罵橁囟娈覺堁矖蘻莥梖昇幉蔕廁趽煉繳坩琣蒄甼冲欘籛堓暂侑勯偌貆燃瘎繥皹密匭氻磌党嶊溹灓梹纛荟絴覄絅匠捕侉瓟狦猣痈衁溣獓磙盖宍椼罤剥瀞摴漑牛肸呇噟嗳搷緃胏炀一㴅"
# rand_spk = "蘁淰教欀射暤忋拍溍儴璸貢炴寅礂垱眣杩覷僂慞礬怞宊觅埆莹桦聟彣立巢猜璶澈擼侮臱聱蛗泩慌澖扱蘕焲溜吔喽穌跉萂咥祉蟬槶挬藃倫稔藼咀柣橁廎煋媵嗨縥哢崘吉姲占屃嵐瓏誱厔伱螖瀴矺啺潽狇寲嚚謱勺穨蚚渞敩嵄瑑爂跰厗琢瞉縄稅榱蚂浭疻蟩熈狳突爖儗朗猙梩芰劻树蠇樐涀嚿蜥圖峌绹廗幃偫佋裛嶺瓇蜬枴沯秐苔昽剩浄媏淸橥擢俧蔥堫諶矯国忳耟流帮菁届仗緐蓄硹嫻漢语栣艋侣沮泙熾唜港曊俟縹虃啷裝赆巬膿渏嫕椉廑嫽斍蔕葇琳畠揪戍塂腯岲蠎様刮拉罜慝袊啒獻窧恊砇唏碐櫡剮藷摽跩嶽玝坚涞窟矬翁茢襳侥甌撰庀詘瓰换它槦栰朱玪勶渜奿摦晞嬼乎嘩冎肵猚旸堺拟嶰淣儙橰圥棅凜裢拡橯謡檯昇孑扸崫収柵犻訣緵蚴彽趬誁岃贤搄蘏矢蟼仒抴紖攉芈衭佑稊旉奨螹疋琞發婙昐肅潯僰禼唬簆嵸疶乳棁犼宒墤幼膡脧尵諲挿砫峓凑嵶膭犯囚詌兾毜伫甬沂価噰煙汳廉昊沱呤俹膣瀙沵桝绦諀贡矮勍翐珿佄胫污臱皗賘境塚尶弛螖捹倕痶暡詡答壈孿凌螨泆捗渼溙奪椻亠憜猌勽厔稨綏蠺吗磝磝嵘媪嶾堇擤脛似脼瀆縟脇喰櫿嘴夙讜嗖癑權挿矸溽哢藬薦蕣蝼漐矘眘凔梒思涞臉寝覛畮望茻嵇篇歛財坹耰拢萞蚙橏泝湸急諱膙猞撣晖揬藊嚰褁硔翮殈臬椖旰禚孳艖惔諧椦舥授搻佘码艨簴勪矙火蔨恞晝櫭囻舡絋嚩瀈臠縅嚒擺悞楿讪襢幌盝庝暠咧笾唸煺螏秭詴杇浔場欜汫泌眈賀譓僇桰耴佽眷賃擏哖差匷磷术淈僵圳蔾欦菖埛楆暨卢凃四櫹煤紇引綧皍傡繄尢虴硿簤滢蓓痓峊甚瞶樼揗慐糪詣平昞萌裛试舐琘愭貪萭吔缜呹紫昷佇敽励擨磦聽杔秸泍凥燾凫熹劅氦墣滿痠枲悏距悷緯簭峹苬敶扄紋壦荂烪嗰漏瀧藐繗碫搼誵燓押瑈嗖潫絸緱疼帆誷蒼熞表罔胂巉贩熊諼楪數巘汑秖狯嬶珶賟忷婿犀槭虈俊塵蜏璄浳笸敳皁檷弼淝擁艡螑裦瀷瀽偅詐缣啟洝碧瓡覝僳祩即橿膴圧玳呣檤茯殫埆糗蟤悌瑼搚妆峢殅珡班泌姼烈职碝稚垥螼盞献棈溋壂昄蝏尟絢玄亽衯潈聁璥蝄忏翵舗绒嘔吪弆詍萅莐攦拝両蒀睿婐考嘭坂覰嘏袇唘谱呢恧简垚曬安蜃樨摫場覰燻覰憤罚号圌紷讑槱筳僻刦哙峎岂碯筛椭櫄倶壤巾晽蔨殠溰墇圳撌妙炩杕掞薗摆豔朑伂杻績背訒毯缎呩惠蒲纂縵襚譲蜟佬侈濃狅婃乚焘羚岌簡誊暗湀寳珦自纥茤格蚿憭綏磭儙俈劘熜蘸紏搙儖袞坩灣蘯僭桰棿芣幀礝怜涨劤充箷沾犮狐跚垝絸耇攇癮絰一㴄"
rand_spk = "蘁淰敡欀尤扸尅綐淄朢炊共諴樶瀁婿秀煵茠湎溾縟犐牊萨蔆痳臣瘛暌汌良褓皉箊柁奶滥淨叓侜璐瓕愭眗攞巘蝼湧恅湂癖粯葻虭浜姯摙癄觝佥撟儵渪凊搊怷裡埳觽斻礅磏舯诽夤榋摍繮觎蜟敋怭伉姦稻彪臹讂榠礫絈捥堀僒凼糌嶱膥狊膎慝櫸蛅溪萖瀓惗穞汊撌调柭豏觰棧足盢悷庪罽覉偼凍仒纴篱亝夯箝讝盆橱漳跹愤牷梷襮欀憑縟臐欀綺盅犴蒕澡亊啦菉犰瞋伒洴絒翊剥犰熳糬繞榆賻譩喢杈沙厐砪栦勦嘎他嚍码洓园楣歎導愯绑熹侱薝勶譐牧剭卼憽挻祜娟胁娛禢圼覥俧奥疪嗟瞸犂硌瘋濱蓼崁姫梅诐倔妅矦瑖屙湻珮渮慔廹繃懖紺晓嘐湏沏莉啯撷冚寍筐礣碃此嗨喓蘥暏蜰挕苹媊茤薍婴棙庠滤僔虩葅弐碒脮玞艝帘莨亭剌螙毯倝崐朘洄欪沎赩莏禑啻焨悷廰噇糩埻嵹拶呠掓窸乭汎幟柗伺娑籰牔壧帔嗑哠瑓腟喜僝煌藩嗝巰椡湀笳泸咔荑臜腢彃碻承殓汘竗壥猵椠沜翖杴縗唺窞和槊毐梗繮先痠橺宕蘁昢徝灊檭矲犄殸税廧椷悄倶獠癚猽攟抹穾置膰筈確豊嗝滥谦櫾甆撝蓷徐灗埸蘩晠灝舖狤脋臥伐瀋懕甪媤瘫浹氏廑蔊眊兺繱牏诠睰瓁怭賺衽嚕懘攰憙蠞殽篺藼潠螆繂旈瓷営尚護汲般摜谢懍秄侏濅敗詞抇尩匸敕誎濯煗兢耓烋蚴訟柳趨溾砼賄肤菊舼伛类撨哿奴毂厏怙繝挽娺卷庽徤慹嵖曰僪蜆咈喾极呭圿煥弒厘萹寶搷滟諰翅杫泸榅橠癤臤礲瞪谞識槮瑷測畋濁籊菓攻浡勏儧艰讵俱哤瘦暃涯緭滯檝嵀旇瓰儝泖綅娟奒晢蒒脨娫玍蛻攗譇耡奿樆帎廹莸絩塦怨滛摽虒产忬腵諠牣彸娃讷廥甥祪勐诊弭傤縨淄嵴剕謍瑰侹祳睲忚烮淔攠赇蜁尴嵮旻僡笋谠喰弸曨茕磷褭聞傩磒貘耞萅悽莨濛捍簨赌茕擛汾噖襼蕾坠殆恺蓚洋諮偄褬佲侥佾屛恋拹偓奇秆儬粧榃灄瓰亓贐娄渰湚孓圬弫妫愛滻榤稲榙散犜糾硙箁惤紓覥刖赯搽燴偬紈拘甐翂櫀筶澬螣砡恫摣絁壧傫桂洷箢涖漂茏壗崭娱亅縛叅利幈烖讬啣羴祚箟奢巆絨荾切観攦猨暁歘懬資崜摰瞂独愐率櫅荰匃艍或即弡獐欞寄杇焯裍擿炑幩忈恰厇禗搋説宰巍往妠胑廆泩嵚禛统諕炏敌薎吣伷晧蒊巤湰偓奉汌営膆榤嘿洔垏娡嫦笖偃藷怋怯十溔薚昬蒌凋惰揈嫫起卤佖萠檘蓤卮爇屚抳訶呺豢曟磐舦讠譨睿衧猭報哬裫孺槹莤諴疚甃拎想菐弶衼詰蟱碊耇櫧謚蓊毎怓瀥棒愐耞窯挠挣戃汁吝燪牬覎笉椀纁枒磮潄山擶睉毒爌仏嚼蔲贫皋帽蚋嘗胧癁悫槈宱潍撳俘婪荙簨湞棄樜一㴆"
params_infer_code = ChatTTS.Chat.InferCodeParams(
    spk_emb = rand_spk, # add sampled speaker 
    temperature = .3,   # using custom temperature
    top_P = 0.7,        # top P decode
    top_K = 20,         # top K decode
)
params_refine_text = ChatTTS.Chat.RefineTextParams(
    prompt='[oral_2][laugh_0][break_6]',
)

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
ans = pipeline(
    Tasks.acoustic_noise_suppression,
    model='iic/speech_frcrn_ans_cirm_16k'
)


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("cosyvoice-api")


# 配置参数
MAX_TEXT_LENGTH = 2000  # 单次请求最大文本长度
CHUNK_SIZE = 30  # 分块大小，确保单次处理的文本不超过30字符
DEFAULT_STYLE = "希望你以后能够做的比我还好呦。"  # 默认语音风格
CACHE_DIR = "cache"  # 缓存目录
CHUANXING_BATCH_SIZE = 64  # 串行批处理大小（超参数）- 串行处理的句子数量
BINGXING_BATCH_SIZE = 8  # 并行批处理大小（超参数）- 同时处理的句子数量


# 创建必要的目录
OUTPUT_DIR = "output"
UPLOAD_DIR = "uploads"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# 配置线程池
thread_pool_executor = concurrent.futures.ThreadPoolExecutor(max_workers=BINGXING_BATCH_SIZE)


    
app = FastAPI(
    title="CosyVoice2 TTS API",
    description="语音合成API服务，支持长文本",
    version="1.0.0",
    # lifespan=lifespan
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# 挂载静态文件目录，使输出文件可以直接通过URL访问
app.mount("/audio", StaticFiles(directory=OUTPUT_DIR), name="audio")


def split_text_bake(text: str) -> List[str]:
    """智能分割文本为句子，确保每段不超过CHUNK_SIZE字符"""
    # 处理中文和英文的标点符号
    sentences = re.split(r'([。！？；：\.!\?;:])', text)
    result = []
    
    # 将分割的标点符号重新附加到前一个分段
    for i in range(0, len(sentences) - 1, 2):
        if i + 1 < len(sentences):
            result.append(sentences[i] + sentences[i + 1])
        else:
            result.append(sentences[i])
    
    # 如果最后一个元素不是标点符号，添加它
    if len(sentences) % 2 == 1:
        result.append(sentences[-1])
        
    # 过滤掉空字符串，并处理过长的句子
    processed = []
    for sentence in result:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # 处理过长的句子，先按逗号等次要标点符号分割
        if len(sentence) > CHUNK_SIZE:
            subparts = re.split(r'([，、,])', sentence)
            subresult = []
            
            for j in range(0, len(subparts) - 1, 2):
                if j + 1 < len(subparts):
                    subresult.append(subparts[j] + subparts[j + 1])
                else:
                    subresult.append(subparts[j])
                    
            if len(subparts) % 2 == 1:
                subresult.append(subparts[-1])
                
            # 进一步处理仍然过长的片段
            for subsentence in subresult:
                if len(subsentence) > CHUNK_SIZE:
                    # 按字符硬切分
                    for k in range(0, len(subsentence), CHUNK_SIZE):
                        chunk = subsentence[k:k+CHUNK_SIZE]
                        if chunk:
                            processed.append(chunk)
                else:
                    processed.append(subsentence)
        else:
            processed.append(sentence)
    
    return processed




def split_text_bake_20250416(text: str, max_size: int = 50) -> List[str]:
    """
    智能分割文本为句子，确保尽可能不拆分完整句子
    
    Args:
        text: 要分割的文本
        max_size: 建议的最大字符数，但完整句子会被保留
    
    Returns:
        分割后的句子列表
    """
    # 主要标点符号分割（这些标点表示句子的结束）
    main_punct_pattern = r'([。！？\.!\?])'
    sentences = re.split(main_punct_pattern, text)
    
    # 将标点符号重新附加到句子
    complete_sentences = []
    i = 0
    while i < len(sentences):
        if i + 1 < len(sentences) and re.match(main_punct_pattern, sentences[i+1]):
            # 当前片段 + 标点符号
            complete_sentences.append(sentences[i] + sentences[i+1])
            i += 2
        else:
            # 没有标点的片段
            if sentences[i].strip():
                complete_sentences.append(sentences[i])
            i += 1
    
    # 次要标点符号（分号、冒号等）作为合并的考虑因素
    secondary_punct_pattern = r'([；：;:])'
    refined_sentences = []
    
    for sentence in complete_sentences:
        if not sentence.strip():
            continue
            
        # 如果句子较短，直接添加
        if len(sentence) <= max_size:
            refined_sentences.append(sentence)
        else:
            # 尝试按次要标点分割长句
            subparts = re.split(secondary_punct_pattern, sentence)
            
            temp_parts = []
            i = 0
            while i < len(subparts):
                if i + 1 < len(subparts) and re.match(secondary_punct_pattern, subparts[i+1]):
                    # 当前片段 + 标点符号
                    temp_parts.append(subparts[i] + subparts[i+1])
                    i += 2
                else:
                    if subparts[i].strip():
                        temp_parts.append(subparts[i])
                    i += 1
            
            # 保留这些分割
            refined_sentences.extend(temp_parts)
    
    # 尝试组合短句，但保留长句
    final_result = []
    current_chunk = ""
    
    for sentence in refined_sentences:
        # 如果句子本身已经超过最大长度，直接保留
        if len(sentence) > max_size:
            if current_chunk:
                final_result.append(current_chunk)
                current_chunk = ""
            final_result.append(sentence)
        # 否则尝试组合
        elif len(current_chunk) + len(sentence) <= max_size:
            current_chunk += sentence
        else:
            if current_chunk:
                final_result.append(current_chunk)
            current_chunk = sentence
    
    # 添加最后一个组合
    if current_chunk:
        final_result.append(current_chunk)
    
    return final_result



def split_text(text: str, max_size: int = 50) -> List[str]:
    """
    智能分割文本为句子，确保尽可能不拆分完整句子
    
    Args:
        text: 要分割的文本
        max_size: 建议的最大字符数，但完整句子会被保留
    
    Returns:
        分割后的句子列表
    """
    # 主要标点符号分割（这些标点表示句子的结束）
    main_punct_pattern = r'([。！？\.!\?])'
    sentences = re.split(main_punct_pattern, text)
    
    # 将标点符号重新附加到句子
    complete_sentences = []
    i = 0
    while i < len(sentences):
        if i + 1 < len(sentences) and re.match(main_punct_pattern, sentences[i+1]):
            # 当前片段 + 标点符号
            complete_sentences.append(sentences[i] + sentences[i+1])
            i += 2
        else:
            # 没有标点的片段
            if sentences[i].strip():
                complete_sentences.append(sentences[i])
            i += 1
    
    # 次要标点符号（分号、冒号等）作为合并的考虑因素
    secondary_punct_pattern = r'([；：;:](?!\d))'  # 添加负向前瞻，避免匹配时间中的冒号
    refined_sentences = []
    
    for sentence in complete_sentences:
        if not sentence.strip():
            continue
        
        # 如果句子较短，直接添加
        if len(sentence) <= max_size:
            refined_sentences.append(sentence)
        else:
            # 尝试按次要标点分割长句
            subparts = re.split(secondary_punct_pattern, sentence)
            
            temp_parts = []
            i = 0
            while i < len(subparts):
                if i + 1 < len(subparts) and re.match(secondary_punct_pattern, subparts[i+1]):
                    # 当前片段 + 标点符号
                    temp_parts.append(subparts[i] + subparts[i+1])
                    i += 2
                else:
                    if subparts[i].strip():
                        temp_parts.append(subparts[i])
                    i += 1
            
            # 保留这些分割
            refined_sentences.extend(temp_parts)
    
    # 尝试组合短句，但保留长句
    final_result = []
    current_chunk = ""
    
    for sentence in refined_sentences:
        # 如果句子本身已经超过最大长度，直接保留
        if len(sentence) > max_size:
            if current_chunk:
                final_result.append(current_chunk)
                current_chunk = ""
            final_result.append(sentence)
        # 否则尝试组合
        elif len(current_chunk) + len(sentence) <= max_size:
            current_chunk += sentence
        else:
            if current_chunk:
                final_result.append(current_chunk)
            current_chunk = sentence
    
    # 添加最后一个组合
    if current_chunk:
        final_result.append(current_chunk)
    
    return final_result



def digits_to_chinese(s, use_yao=False):
    num_map = {
        '0': '零',
        '1': '幺' if use_yao else '一',
        '2': '二',
        '3': '三',
        '4': '四',
        '5': '五',
        '6': '六',
        '7': '七',
        '8': '八',
        '9': '九',
    }
    return ''.join(num_map[c] for c in s)

def minute_to_chinese(mm):
    if mm == '00':
        return '整'
    elif mm.startswith('0'):
        return '零' + cn2an.an2cn(int(mm[1])) + '分'
    elif mm == '10':
        return '十分'
    else:
        return cn2an.an2cn(int(mm)) + '分'

def time_range_repl(m):
    h1, m1 = m.group(1), m.group(2)
    h2, m2 = m.group(3), m.group(4)
    left = cn2an.an2cn(int(h1)) + '点' + minute_to_chinese(m1)
    right = cn2an.an2cn(int(h2)) + '点' + minute_to_chinese(m2)
    return f"{left}至{right}"

def time_repl(m):
    h, mi = m.group(1), m.group(2)
    h_txt = cn2an.an2cn(int(h))
    mi_txt = minute_to_chinese(mi)
    return f"{h_txt}点{mi_txt}"

def preprocess_chattts_text(sentence: str):
    if not sentence:
        return ""
    
    units = ["分钟", "小时", "秒", "天", "周", "月", "年", "次", "千米", "米", "厘米", 
             "毫米", "公里", "千克", "克", "毫克", "吨", "升", "毫升", "度", "摄氏度", "华氏度", "%"]

    # 1. 区号-手机号：加特殊TAG防止二次转换
    def phone_with_code_repl(m):
        code = m.group(1)
        number = m.group(2)
        # 用<tag>包裹区号部分，防止后续被改写
        code_cn = digits_to_chinese(code)
        number_cn = digits_to_chinese(number, use_yao=True)
        return f"<code>{code_cn}</code>-{number_cn}"
    sentence = re.sub(r'\b(\d{2,3})[-—](1\d{10})\b', phone_with_code_repl, sentence)
    sentence = re.sub(r'\+\s?(\d{2,3})[-—](1\d{10})\b', phone_with_code_repl, sentence)

    # 2. 处理11位手机号（非区号场景）
    def phone_repl(m):
        return digits_to_chinese(m.group(0), use_yao=True)
    sentence = re.sub(r'(?<!\d)(1\d{10})(?!\d)', phone_repl, sentence)

    # 3. 处理年份，逐位读
    sentence = re.sub(
        r'(\d{4})年',
        lambda m: digits_to_chinese(m.group(1)) + '年',
        sentence,
    )

    # 4. 处理时间区间
    sentence = re.sub(
        r'(\d{1,2}):(\d{2})\s*[-~—]\s*(\d{1,2}):(\d{2})',
        time_range_repl, sentence
    )

    # 5. 处理单个时间
    sentence = re.sub(
        r'(\d{1,2}):(\d{2})(?!\d)', time_repl, sentence
    )

    # 6. 百分比
    sentence = re.sub(r'(\d+)%', lambda m: '百分之' + cn2an.an2cn(int(m.group(1))), sentence)

    # 7. 比例表达
    sentence = re.sub(r'(\d+)\s*:\s*(\d+)', lambda m: cn2an.an2cn(int(m.group(1))) + '比' + cn2an.an2cn(int(m.group(2))), sentence)

    # 8. 单位表达
    for unit1 in units:
        for unit2 in units:
            pattern = rf'(\d+)\s*{unit1}/({unit2})'
            sentence = re.sub(pattern, lambda m: cn2an.an2cn(int(m.group(1))) + unit1 + '每' + m.group(2), sentence)

    sentence = re.sub(r'(\d+)\s*[°℃]\s*C?', lambda m: cn2an.an2cn(int(m.group(1))) + '摄氏度', sentence)
    sentence = re.sub(r'(\d+)\s*[°℉]\s*F?', lambda m: cn2an.an2cn(int(m.group(1))) + '华氏度', sentence)

    for unit in units:
        pattern = rf'(\d+)\s*{unit}'
        sentence = re.sub(pattern, lambda m: cn2an.an2cn(int(m.group(1))) + unit, sentence)

    # 9. 日期处理
    sentence = re.sub(
        r'(\d{4})-(\d{1,2})-(\d{1,2})',
        lambda m: digits_to_chinese(m.group(1)) + '年' + cn2an.an2cn(int(m.group(2))) + '月' + cn2an.an2cn(int(m.group(3))) + '日',
        sentence
    )
    sentence = re.sub(
        r'(\d{4})/(\d{1,2})/(\d{1,2})',
        lambda m: digits_to_chinese(m.group(1)) + '年' + cn2an.an2cn(int(m.group(2))) + '月' + cn2an.an2cn(int(m.group(3))) + '日',
        sentence
    )
    sentence = re.sub(
        r'(\d{1,2})/(\d{1,2})(?!/)',
        lambda m: cn2an.an2cn(int(m.group(1))) + '月' + cn2an.an2cn(int(m.group(2))) + '日',
        sentence
    )

    # 10. 标点和特殊符号转换为空格
    pattern = r'[^\u4e00-\u9fa5a-zA-Z0-9<>/-]'
    def replace_with_space(match):
        matched_text = match.group(0)
        for unit in units:
            if matched_text in unit:
                return matched_text
        return " "
    sentence = re.sub(pattern, replace_with_space, sentence)
    
    # 11. 剩余数字转为汉字（保护带有<tag>的区号）
    # 只替换未被<tag>包裹的数字
    def safe_cn2an(m):
        # 如果在<code>标签内则跳过
        if m.group(0).startswith('<code>') and m.group(0).endswith('</code>'):
            return m.group(0)
        else:
            return cn2an.an2cn(int(m.group(0)))
    # 先把<code>部分临时替换为特殊字符，避免被正则命中
    code_list = []
    def code_save(m):
        code_list.append(m.group(0))
        return f'[[[CODE{len(code_list)-1}]]]'
    sentence = re.sub(r'<code>.*?</code>', code_save, sentence)
    # 替换剩余数字
    sentence = re.sub(r'\d+', lambda m: cn2an.an2cn(int(m.group(0))), sentence)
    # 恢复<code>部分
    def code_restore(m):
        idx = int(m.group(1))
        # 去掉<code>标签
        return code_list[idx][6:-7]
    sentence = re.sub(r'\[\[\[CODE(\d+)\]\]\]', code_restore, sentence)
    
    sentence = re.sub(r'\s+', ' ', sentence)
    return sentence.strip()



def preprocess_chattts_text_bake_20250416(sentence: str):
    """
    对输入字符串进行处理，使其适合TTS朗读
    
    处理内容：
    1. 将特殊符号和标点符号转为空格
    2. 将数字转为汉字表示
    3. 将日期格式转为汉字表示
    4. 处理特殊表达方式
    
    Args:
        sentence: 输入字符串
    
    Returns:
        处理后的字符串
    """
    if not sentence:
        return ""
    
    # 保存单位词汇，避免被替换
    units = ["分钟", "小时", "秒", "天", "周", "月", "年", "次", "千米", "米", "厘米", 
             "毫米", "公里", "千克", "克", "毫克", "吨", "升", "毫升", "度", "摄氏度", "华氏度", "%"]
    
    # 步骤1：特殊表达式处理 (在标点符号转换前进行)
    
    # 处理百分比
    sentence = re.sub(r'(\d+)%', lambda m: '百分之' + cn2an.an2cn(int(m.group(1))), sentence)
    
    # 处理比例表达 (如 1:2, 3:4)
    sentence = re.sub(r'(\d+)\s*:\s*(\d+)', lambda m: cn2an.an2cn(int(m.group(1))) + '比' + cn2an.an2cn(int(m.group(2))), sentence)
    
    # 处理单位表达 (如 2次/分钟)
    # 首先处理形如 "数字+单位/单位" 的模式
    for unit1 in units:
        for unit2 in units:
            pattern = rf'(\d+)\s*{unit1}/({unit2})'
            sentence = re.sub(pattern, lambda m: cn2an.an2cn(int(m.group(1))) + unit1 + '每' + m.group(2), sentence)
    
    # 处理摄氏度等特殊单位
    sentence = re.sub(r'(\d+)\s*[°℃]\s*C?', lambda m: cn2an.an2cn(int(m.group(1))) + '摄氏度', sentence)
    sentence = re.sub(r'(\d+)\s*[°℉]\s*F?', lambda m: cn2an.an2cn(int(m.group(1))) + '华氏度', sentence)
    
    # 处理一般的数字+单位形式
    for unit in units:
        pattern = rf'(\d+)\s*{unit}'
        sentence = re.sub(pattern, lambda m: cn2an.an2cn(int(m.group(1))) + unit, sentence)
    
    # 步骤2：日期处理 (常见格式如 YYYY-MM-DD, YYYY/MM/DD, MM/DD等)
    # 年-月-日格式
    sentence = re.sub(r'(\d{4})-(\d{1,2})-(\d{1,2})', 
                      lambda m: cn2an.an2cn(int(m.group(1))) + '年' + 
                                cn2an.an2cn(int(m.group(2))) + '月' + 
                                cn2an.an2cn(int(m.group(3))) + '日', 
                      sentence)
    
    # 年/月/日格式
    sentence = re.sub(r'(\d{4})/(\d{1,2})/(\d{1,2})', 
                      lambda m: cn2an.an2cn(int(m.group(1))) + '年' + 
                                cn2an.an2cn(int(m.group(2))) + '月' + 
                                cn2an.an2cn(int(m.group(3))) + '日', 
                      sentence)
    
    # 月/日格式
    sentence = re.sub(r'(\d{1,2})/(\d{1,2})(?!/)', 
                      lambda m: cn2an.an2cn(int(m.group(1))) + '月' + 
                                cn2an.an2cn(int(m.group(2))) + '日', 
                      sentence)
    
    # 步骤3：标点和特殊符号转换为空格
    # 创建一个不包含中文字符、字母、数字和单位的模式
    pattern = r'[^\u4e00-\u9fa5a-zA-Z0-9]'
    
    # 将标点和特殊符号替换为空格，但保留单位
    def replace_with_space(match):
        matched_text = match.group(0)
        # 检查是否是单位的一部分
        for unit in units:
            if matched_text in unit:
                return matched_text
        return " "
    
    sentence = re.sub(pattern, replace_with_space, sentence)
    
    # 步骤4：将剩余数字转为汉字
    sentence = re.sub(r'\d+', lambda m: cn2an.an2cn(int(m.group(0))), sentence)
    
    # 替换多个空格为单个空格
    sentence = re.sub(r'\s+', ' ', sentence)
    
    return sentence.strip()





def preprocess_chattts_text_bake(
    sentence: str   
    ):
    """
    对输入字符串进行处理，使其适合TTS朗读
    
    处理内容：
    1. 将特殊符号和标点符号转为空格
    2. 将数字转为汉字表示
    3. 将日期格式转为汉字表示
    4. 处理特殊表达方式
    
    Args:
        sentence: 输入字符串
    
    Returns:
        处理后的字符串
    """
    if not sentence:
        return ""
    
    # 保存单位词汇，避免被替换
    units = ["分钟", "小时", "秒", "天", "周", "月", "年", "次", "千米", "米", "厘米", 
             "毫米", "公里", "千克", "克", "毫克", "吨", "升", "毫升", "度", "摄氏度", "华氏度"]
    
    # 步骤1：特殊表达式处理 (在标点符号转换前进行)
    
    # 处理比例表达 (如 1:2, 3:4)
    sentence = re.sub(r'(\d+)\s*:\s*(\d+)', lambda m: cn2an.an2cn(int(m.group(1))) + '比' + cn2an.an2cn(int(m.group(2))), sentence)
    
    # 处理单位表达 (如 2次/分钟)
    # 寻找"数字+单位/单位"的模式
    for unit1 in units:
        for unit2 in units:
            pattern = rf'(\d+)({unit1})/({unit2})'
            sentence = re.sub(pattern, lambda m: cn2an.an2cn(int(m.group(1))) + m.group(2) + '每' + m.group(3), sentence)
    
    # 步骤2：标点和特殊符号转换为空格
    # 保留字母、汉字、单位，其他转为空格
    # 创建模式排除字母和单位
    pattern = r'[^\u4e00-\u9fa5a-zA-Z0-9'
    for unit in units:
        pattern += unit.replace("", "|")
    pattern += r']'
    
    # 将标识的标点和特殊符号替换为空格
    # 注意：这种方法可能有限制，如果单位被拆分可能会误处理
    def replace_with_space(match):
        if any(unit in match.group(0) for unit in units):
            return match.group(0)
        else:
            return " "
    
    sentence = re.sub(pattern, replace_with_space, sentence)
    
    # 步骤3：日期处理 (常见格式如 YYYY-MM-DD, YYYY/MM/DD, MM/DD等)
    # 年-月-日格式
    sentence = re.sub(r'(\d{4})-(\d{1,2})-(\d{1,2})', 
                      lambda m: cn2an.an2cn(int(m.group(1))) + '年' + 
                                cn2an.an2cn(int(m.group(2))) + '月' + 
                                cn2an.an2cn(int(m.group(3))) + '日', 
                      sentence)
    
    # 年/月/日格式
    sentence = re.sub(r'(\d{4})/(\d{1,2})/(\d{1,2})', 
                      lambda m: cn2an.an2cn(int(m.group(1))) + '年' + 
                                cn2an.an2cn(int(m.group(2))) + '月' + 
                                cn2an.an2cn(int(m.group(3))) + '日', 
                      sentence)
    
    # 月/日格式
    sentence = re.sub(r'(\d{1,2})/(\d{1,2})(?!/)', 
                      lambda m: cn2an.an2cn(int(m.group(1))) + '月' + 
                                cn2an.an2cn(int(m.group(2))) + '日', 
                      sentence)
    
    # 步骤4：将剩余数字转为汉字
    # 使用正则表达式匹配数字并转换
    sentence = re.sub(r'\d+', lambda m: cn2an.an2cn(int(m.group(0))), sentence)
    
    # 替换多个空格为单个空格
    sentence = re.sub(r'\s+', ' ', sentence)
    
    return sentence.strip()


def process_sentence_sync(
    sentence: str,
    sentence_idx: int,
    prompt_speech: torch.Tensor,
    style: str = DEFAULT_STYLE,
    instruct: Optional[str] = None,
    stream: bool = False,
    speed: float = 1.0
) -> Tuple[int, List[Dict[str, Any]]]:
    """
    同步处理单个句子（用于在线程池中执行）
    """
    try:
        logger.info(f"处理句子 (索引: {sentence_idx}): '{sentence}'")
        text = [preprocess_chattts_text(item) for item in sentence]
        logger.info(f"text: ------------------------- {text}")
        results = chat.infer(
            text,
            params_refine_text=params_refine_text,
            params_infer_code=params_infer_code,
        )
        
        return sentence_idx, results
    except Exception as e:
        logger.error(f"处理句子 '{sentence}' (索引: {sentence_idx}) 时出错: {str(e)}")
        return sentence_idx, []  # 返回空结果列表表示处理失败


async def process_sentence_batch_multi_thread(
    batch_sentences: List[str],
    batch_indices: List[int],
    prompt_speech: torch.Tensor,
    style: str = # The above code is a comment in Python. Comments are used to provide explanations or
    # notes within the code for better understanding. In this case, the comment appears
    # to be indicating a default style setting.
    DEFAULT_STYLE,
    instruct: Optional[str] = None,
    stream: bool = False,
    speed: float = 1.0
) -> List[Tuple[int, Dict[str, Any]]]:
    """使用多线程并行处理一批文本句子并返回结果"""


    batch_results = []
    loop = asyncio.get_event_loop()
    
    # 使用并行批次做进一步拆分
    bingxing_batches = []
    bingxing_batch_indices = []
    for i in range(0, len(batch_sentences), BINGXING_BATCH_SIZE):
        batch = batch_sentences[i:i+BINGXING_BATCH_SIZE]
        indices = list(range(i, min(i+BINGXING_BATCH_SIZE, len(batch_sentences))))
        bingxing_batches.append(batch)
        bingxing_batch_indices.append(indices)
    
    
    
    # 准备多线程任务
    futures = []
    for i, (sentence_idx, sentence) in enumerate(zip(bingxing_batch_indices, bingxing_batches)):
        future = loop.run_in_executor(
            thread_pool_executor,
            process_sentence_sync,
            sentence,
            sentence_idx,
            prompt_speech,
            style,
            instruct,
            stream,
            speed
        )
        futures.append(future)
    
    # 等待所有任务完成
    results = await asyncio.gather(*futures)
    
    # 处理结果
    for sentence_idx, sentence_results in results:
        for result in sentence_results:
            batch_results.append((sentence_idx, result))
    
    return batch_results


async def process_sentence_batch(
    batch_sentences: List[str],
    batch_indices: List[int],
    prompt_speech: torch.Tensor,
    style: str = DEFAULT_STYLE,
    instruct: Optional[str] = None,
    stream: bool = False,
    speed: float = 1.0
) -> List[Tuple[int, Dict[str, Any]]]:
    """并行处理一批文本句子并返回结果"""
    # 创建任务列表
    tasks = []
    for i, (sentence_idx, sentence) in enumerate(zip(batch_indices, batch_sentences)):
        # 为每个句子创建异步任务
        task = process_sentence(
            sentence, sentence_idx, prompt_speech, style, instruct, stream, speed
        )
        tasks.append(task)
    
    # 并行执行所有任务
    batch_results = []
    results = await asyncio.gather(*tasks)
    
    # 处理结果
    for sentence_idx, sentence_results in results:
        # 将每个句子的结果与其索引一起添加到批次结果中
        for result in sentence_results:
            batch_results.append((sentence_idx, result))
    
    return batch_results


async def process_long_text_batch(
    text: str, 
    prompt_speech: torch.Tensor, 
    style: str = DEFAULT_STYLE,
    instruct: Optional[str] = None,
    stream: bool = False,
    speed: float = 1.0
) -> List[Dict[str, Any]]:
    """处理长文本并返回结果列表，使用批处理并行处理"""
    if text is None:
        raise ValueError("文本和提示音频不能为空")
        
    # 检查模型是否已加载
    # if cosyvoice is None:
    #     await load_model()
    #     if cosyvoice is None:
    #         raise HTTPException(status_code=503, detail="TTS模型未加载")
    
    # 分割长文本为小片段
    sentences = split_text(text)
    
    logger.info(f"文本已分割为 {len(sentences)} 个片段，使用批处理大小: {BINGXING_BATCH_SIZE}")
    
    # 创建批次
    batches = []
    batch_indices = []
    for i in range(0, len(sentences), CHUANXING_BATCH_SIZE):
        batch = sentences[i:i+CHUANXING_BATCH_SIZE]
        indices = list(range(i, min(i+CHUANXING_BATCH_SIZE, len(sentences))))
        batches.append(batch)
        batch_indices.append(indices)
    
    logger.info(f"创建了 {len(batches)} 个批次进行处理")
    
    # 存储所有结果（带索引，以确保正确顺序）
    all_indexed_results = []
    
    # 逐批处理
    for batch_num, (batch, indices) in enumerate(zip(batches, batch_indices)):
        try:
            logger.info(f"开始处理第 {batch_num+1}/{len(batches)} 批句子")
            
            # 处理当前批次（多线程并行）
            batch_results = await process_sentence_batch_multi_thread(
                batch, indices, prompt_speech, style, instruct, stream, speed
            )
            
            # 添加到结果列表
            all_indexed_results.extend(batch_results)
            
            # 清理显存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            logger.error(f"处理第 {batch_num+1} 批次时出错: {str(e)}")
            # 继续处理其他批次
    
    # 按原始索引排序结果，确保正确顺序
    all_indexed_results.sort(key=lambda x: x[0])
    
    # 返回排序后的结果列表（仅保留结果部分，丢弃索引）
    return [result for _, result in all_indexed_results]



def optimize_speech_processing(all_speeches: List[np.ndarray], silence_duration: float = 0.05, trim_threshold: float = 0.01) -> np.ndarray:
    """
    优化语音片段合并，包括:
    1. 修剪每个片段末尾的静音
    2. 标准化音量
    3. 精确控制片段间间隔
    4. 应用平滑过渡
    
    参数:
    - all_speeches: 所有语音片段列表(numpy数组)
    - silence_duration: 片段之间的静音持续时间(秒)，默认0.05秒
    - trim_threshold: 用于检测静音的振幅阈值，默认0.01
    """
    if not all_speeches:
        return None
    
    # 获取采样率（默认采样率通常是16000Hz）
    sample_rate = 16000
    silence_samples = int(silence_duration * sample_rate)
    
    # 检查和过滤无效数组
    valid_speeches = []
    for speech in all_speeches:
        if isinstance(speech, np.ndarray) and speech.size > 0 and not np.isnan(speech).any() and np.isfinite(speech).all():
            valid_speeches.append(speech)
    
    if not valid_speeches:
        logger.info(f"invalid speech!\n {all_speeches}")
        return None
    
    # 修剪和标准化每个片段
    processed_speeches = []
    for speech in valid_speeches:
        # 确保数组是一维的
        if speech.ndim > 1:
            speech = speech.flatten()
            
        # 标准化音量
        rms = np.sqrt(np.mean(np.square(speech)))
        if rms > 0:
            # 应用标准化，目标RMS为0.05
            target_rms = 0.05
            speech = speech * (target_rms / rms)
        
        # 修剪末尾静音
        # 从末尾向前扫描，找到第一个超过阈值的样本
        abs_speech = np.abs(speech)
        non_silent_pos = np.where(abs_speech > trim_threshold)[0]
        
        if len(non_silent_pos) > 0:
            # 找到最后一个非静音样本的位置
            last_sound_pos = non_silent_pos[-1]
            # 保留一点点尾音（例如50ms）以确保自然度
            tail_samples = int(0.05 * sample_rate)
            end_pos = min(last_sound_pos + tail_samples, len(speech))
            # 修剪片段
            speech = speech[:end_pos]
        
        processed_speeches.append(speech)
    
    # 如果只有一个语音片段，直接返回
    if len(processed_speeches) == 1:
        return processed_speeches[0]
    
    # 创建合并片段，控制间隔时间
    combined_speeches = []
    for i, speech in enumerate(processed_speeches):
        combined_speeches.append(speech)
        
        # 在除最后一个片段外的每个片段后添加指定长度的静音
        if i < len(processed_speeches) - 1:
            silence = np.zeros(silence_samples, dtype=speech.dtype)
            combined_speeches.append(silence)
    
    # 合并所有片段
    all_speech = np.concatenate(combined_speeches)
    
    # 应用跨片段平滑（可选，根据需要）
    # 这部分可能需要根据实际效果调整
    
    return all_speech


def optimize_speech_processing_bake(all_speeches: List[torch.Tensor], trim_threshold: float = 0.01, crossfade_duration: float = 0.03) -> torch.Tensor:
    """
    优化语音片段无缝合并，包括:
    1. 彻底修剪每个片段开头和结尾的静音
    2. 标准化音量
    3. 使用交叉淡入淡出技术实现平滑过渡
    4. 零间隔连接
    
    参数:
    - all_speeches: 所有语音片段列表
    - trim_threshold: 用于检测静音的振幅阈值，默认0.01
    - crossfade_duration: 交叉淡变持续时间(秒)，默认0.03秒
    """
    if not all_speeches:
        return None
    
    # 获取采样率（CosyVoice的默认采样率通常是16000Hz）
    sample_rate = 24000
    crossfade_samples = int(crossfade_duration * sample_rate)
    
    # 检查和过滤无效张量
    valid_speeches = []
    for speech in all_speeches:
        if isinstance(speech, torch.Tensor) and speech.numel() > 0:
            valid_speeches.append(speech)
    
    if not valid_speeches:
        return None
    
    # 修剪和标准化每个片段
    processed_speeches = []
    for speech in valid_speeches:
        # 标准化音量
        rms = torch.sqrt(torch.mean(speech ** 2))
        if rms > 0:
            # 应用标准化，目标RMS为0.05
            target_rms = 0.05
            speech = speech * (target_rms / rms)
        
        # 修剪前后静音
        abs_speech = torch.abs(speech[0])  # 假设是单声道
        
        # 找到所有非静音样本的位置
        non_silent_pos = torch.where(abs_speech > trim_threshold)[0]
        
        if len(non_silent_pos) > 0:
            # 找到第一个和最后一个非静音样本的位置
            first_sound_pos = non_silent_pos[0].item()
            last_sound_pos = non_silent_pos[-1].item()
            
            # 在开头保留少量前导（10ms），以避免切得太死
            start_pos = max(0, first_sound_pos - int(0.01 * sample_rate))
            
            # 在结尾保留少量尾音（20ms），以保持自然感
            end_pos = min(last_sound_pos + int(0.02 * sample_rate), speech.shape[1])
            
            # 修剪片段
            speech = speech[:, start_pos:end_pos]
        
        processed_speeches.append(speech)
    
    # 如果只有一个语音片段，直接返回
    if len(processed_speeches) == 1:
        return processed_speeches[0]
    
    # 使用交叉淡变实现无缝连接
    combined_speech = processed_speeches[0]
    
    for i in range(1, len(processed_speeches)):
        next_speech = processed_speeches[i]
        
        # 确保交叉淡变区域不超过当前片段的长度
        actual_crossfade = min(crossfade_samples, combined_speech.shape[1], next_speech.shape[1])
        
        if actual_crossfade > 0:
            # 创建淡出和淡入曲线
            fade_out = torch.linspace(1, 0, actual_crossfade)
            fade_in = torch.linspace(0, 1, actual_crossfade)
            
            # 应用淡出到当前合并片段的末尾
            combined_end = combined_speech[:, -actual_crossfade:]
            faded_end = combined_end * fade_out
            
            # 应用淡入到下一个片段的开头
            next_start = next_speech[:, :actual_crossfade]
            faded_start = next_start * fade_in
            
            # 混合交叉区域
            crossfade_region = faded_end + faded_start
            
            # 创建新的合并片段，去掉当前片段的交叉区域，添加混合区域和下一个片段的余下部分
            new_combined = torch.cat([
                combined_speech[:, :-actual_crossfade],
                crossfade_region,
                next_speech[:, actual_crossfade:]
            ], dim=1)
            
            combined_speech = new_combined
        else:
            # 如果交叉淡变不可能，直接连接
            combined_speech = torch.cat([combined_speech, next_speech], dim=1)
    
    return combined_speech


async def process_long_text(
    text: str, 
    prompt_speech: torch.Tensor, 
    style: str = DEFAULT_STYLE,
    instruct: Optional[str] = None,
    stream: bool = False,
    speed: float = 1.0
) -> List[Dict[str, Any]]:
    """处理长文本并返回结果列表"""
    if not text or prompt_speech is None:
        raise ValueError("文本和提示音频不能为空")
        
    # 检查模型是否已加载
    if cosyvoice is None:
        await load_model()
        if cosyvoice is None:
            raise HTTPException(status_code=503, detail="TTS模型未加载")
    
    # 分割长文本为小片段
    sentences = split_text(text)
    logger.info(f"文本已分割为 {len(sentences)} 个片段")
    logger.info(f"sentences: {sentences} ")
    
    # 存储所有结果
    all_results = []
    
    # 逐段处理
    for i, sentence in enumerate(sentences):
        try:
            # 清理显存
            if i > 0 and i % 3 == 0:  # 每处理3个片段清理一次显存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            # 使用适当的推理方法
            if instruct:
                results = list(cosyvoice.inference_instruct2(
                    # The above code is not valid Python code. It seems to contain some random text
                    # ("sentence") and comment symbols ("
                    sentence, instruct, prompt_speech, stream=stream, speed=speed
                ))
            else:
                results = list(cosyvoice.inference_zero_shot(
                    sentence, style, prompt_speech, stream=stream, speed=speed
                ))
            
            # 添加到结果列表
            if results and len(results) > 0:
                all_results.extend(results)
                
            # 在控制台输出进度
            logger.info(f"已处理 {i+1}/{len(sentences)} 个片段: '{sentence}'")
            
        except Exception as e:
            logger.error(f"处理第 {i+1} 个文本片段 '{sentence}' 时出错: {str(e)}")
            # 继续处理其他片段
            continue
    
    return all_results


# prompt_speech_16k = load_wav('/work/soft/CosyVoice/asset/zero_shot_prompt.wav', 16000)
prompt_speech_16k = None



# 创建请求模型
class TTSRequest(BaseModel):
    text: str
    style: str = DEFAULT_STYLE
    instruct: Optional[str] = None
    wait_complete: bool = False
    speed: float = 1.0
    use_batch: bool = False


@app.post("/tts")
async def tts(
    request: Request,
    background_tasks: BackgroundTasks,
    tts_request: TTSRequest,
):
    # 在这里处理请求
    text = tts_request.text
    style = tts_request.style
    instruct = tts_request.instruct
    wait_complete = tts_request.wait_complete
    speed = tts_request.speed
    use_batch = tts_request.use_batch
    """长文本语音合成API，可选择异步处理和批处理
    非流式生成，后续添加流式生成，使用generator
    """
    # 验证速度参数
    if speed < 0.5 or speed > 2.0:
        raise HTTPException(status_code=400, detail="速度参数必须在0.5到2.0之间")
    prompt_audio = None
    prompt_path = None
    final_prompt_speech_16k = None
    try:
        # 加载提示音频
        if prompt_audio is not None:
            # 保存上传的音频文件
            prompt_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}.wav")
            with open(prompt_path, "wb") as buffer:
                shutil.copyfileobj(prompt_audio.file, buffer)
            final_prompt_speech_16k = load_wav(prompt_path, 16000)
        else:
            # 使用默认的提示音频
            final_prompt_speech_16k = prompt_speech_16k
        
        # 生成任务ID和输出文件名
        task_id = str(uuid.uuid4())
        output_filename = f"{task_id}.wav"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        # 构建结果URL
        base_url = str(request.base_url).rstrip('/')
        audio_url = f"{base_url}/audio/{output_filename}"
        
        # 定义异步处理函数
        async def process_text_task():
            try:
                # 根据use_batch选择处理方法
                if use_batch:
                    logger.info(f"使用批处理模式处理文本，批处理大小: {BINGXING_BATCH_SIZE}")
                    results = await process_long_text_batch(
                        text, 
                        final_prompt_speech_16k, 
                        style, 
                        instruct, 
                        stream=False, 
                        speed=speed
                    )
                else:
                    logger.info("使用常规模式处理文本（不使用批处理）")
                    results = await process_long_text(
                        text, 
                        final_prompt_speech_16k, 
                        style, 
                        instruct, 
                        stream=False, 
                        speed=speed
                    )
                
                # 合并所有音频片段，使用优化的音频处理
                if results and len(results) > 0:
                    # all_speeches = [result['tts_speech'] for result in results if 'tts_speech' in result]
                    all_speeches = results
                    if all_speeches:
                        # 使用较短的间隔时间，例如0.1秒
                        all_speech = optimize_speech_processing(all_speeches, silence_duration=0.001)
                        if all_speech is not None:
                            try:
                                torchaudio.save(output_path, torch.from_numpy(all_speech).unsqueeze(0), 24000)
                            except:
                                torchaudio.save(output_path, torch.from_numpy(all_speech), 24000)
                            result = ans(
                                output_path,
                                output_path=output_path
                            )
                            # torchaudio.save(output_path, all_speech, cosyvoice.sample_rate)
                            
                            # 保存任务状态
                            with open(os.path.join(CACHE_DIR, f"{task_id}.status"), "w") as f:
                                f.write("completed")
                        else:
                            with open(os.path.join(CACHE_DIR, f"{task_id}.status"), "w") as f:
                                f.write("error: 音频处理失败")
                    else:
                        with open(os.path.join(CACHE_DIR, f"{task_id}.status"), "w") as f:
                            f.write("error: 未生成有效的语音片段")
                else:
                    # 没有结果时保存错误状态
                    with open(os.path.join(CACHE_DIR, f"{task_id}.status"), "w") as f:
                        f.write("error: 未生成有效的TTS结果")
                        
                # 清理显存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                error_info = traceback.format_exc()
                logger.error(f"异步处理任务 {task_id} 时出错: {str(e)}\n调用栈信息: {error_info}")
                # 保存错误状态
                with open(os.path.join(CACHE_DIR, f"{task_id}.status"), "w") as f:
                    f.write(f"error: {str(e)}")
            finally:
                # 清理临时文件
                if prompt_path is not None and os.path.exists(prompt_path):
                    os.remove(prompt_path)
        
        if wait_complete:
            # 同步处理
            await process_text_task()
            return {
                "success": True,
                "task_id": task_id,
                "audio_url": audio_url,
                "status": "completed",
                "text_length": len(text),
                "processed_text": text,
                "speed": speed,
                "use_batch": use_batch,
                "batch_size": BINGXING_BATCH_SIZE if use_batch else None,
                "message": "长文本语音合成成功"
            }
        else:
            # 异步处理
            background_tasks.add_task(process_text_task)
            
            # 初始化任务状态
            with open(os.path.join(CACHE_DIR, f"{task_id}.status"), "w") as f:
                f.write("processing")
                
            return {
                "success": True,
                "task_id": task_id,
                "status": "processing",
                "text_length": len(text),
                "processed_text": text,
                "speed": speed,
                "use_batch": use_batch,
                "batch_size": BINGXING_BATCH_SIZE if use_batch else None,
                "check_status_url": f"{base_url}/status/{task_id}",
                "message": "长文本语音合成任务已开始处理"
            }
    
    except Exception as e:
        logger.error(f"处理长文本请求时出错: {str(e)}")
        # 清理临时文件
        if prompt_path is not None and os.path.exists(prompt_path):
            os.remove(prompt_path)
        raise HTTPException(status_code=500, detail=f"TTS处理错误: {str(e)}")



@app.post("/tts/bake")
async def tts_bake(
    request: Request,
    background_tasks: BackgroundTasks,
    text: str = Form(...),
    prompt_audio: Optional[UploadFile] = File(None),
    style: str = Form(DEFAULT_STYLE),
    instruct: Optional[str] = Form(None),
    wait_complete: bool = Form(False),
    speed: float = Form(1.0)
):
    """长文本语音合成API，可选择异步处理
    非流式生成，后续添加流式生成，使用generator
    """
    # 验证速度参数
    if speed < 0.5 or speed > 2.0:
        raise HTTPException(status_code=400, detail="速度参数必须在0.5到2.0之间")
    
    prompt_path = None
    try:
        # 加载提示音频
        if prompt_audio is not None:
            # 保存上传的音频文件
            prompt_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}.wav")
            with open(prompt_path, "wb") as buffer:
                shutil.copyfileobj(prompt_audio.file, buffer)
            prompt_speech_16k = load_wav(prompt_path, 16000)
        else:
            # 使用默认的提示音频
            prompt_speech_16k = load_wav('/work/soft/CosyVoice/asset/zero_shot_prompt.wav', 16000)
        
        # 生成任务ID和输出文件名
        task_id = str(uuid.uuid4())
        output_filename = f"{task_id}.wav"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        # 构建结果URL
        base_url = str(request.base_url).rstrip('/')
        audio_url = f"{base_url}/audio/{output_filename}"
        # 定义异步处理函数
        async def process_text_task():
            try:
                # 处理文本
                results = await process_long_text(text, prompt_speech_16k, style, instruct, stream=False, speed=speed)
                
                # 合并所有音频片段，使用优化的音频处理
                if results and len(results) > 0:
                    all_speeches = [result['tts_speech'] for result in results if 'tts_speech' in result]
                    if all_speeches:
                        # 使用较短的间隔时间，例如0.1秒
                        all_speech = optimize_speech_processing(all_speeches, silence_duration=0.05)
                        if all_speech is not None:
                            torchaudio.save(output_path, all_speech, cosyvoice.sample_rate)
                            
                            # 保存任务状态
                            with open(os.path.join(CACHE_DIR, f"{task_id}.status"), "w") as f:
                                f.write("completed")
                        else:
                            with open(os.path.join(CACHE_DIR, f"{task_id}.status"), "w") as f:
                                f.write("error: 音频处理失败")
                    else:
                        with open(os.path.join(CACHE_DIR, f"{task_id}.status"), "w") as f:
                            f.write("error: 未生成有效的语音片段")
                else:
                    # 没有结果时保存错误状态
                    with open(os.path.join(CACHE_DIR, f"{task_id}.status"), "w") as f:
                        f.write("error: 未生成有效的TTS结果")
                        
                # 清理显存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                logger.error(f"异步处理任务 {task_id} 时出错: {str(e)}")
                # 保存错误状态
                with open(os.path.join(CACHE_DIR, f"{task_id}.status"), "w") as f:
                    f.write(f"error: {str(e)}")
            finally:
                # 清理临时文件
                if prompt_path is not None and os.path.exists(prompt_path):
                    os.remove(prompt_path)
        
        if wait_complete:
            # 同步处理
            await process_text_task()
            return {
                "success": True,
                "task_id": task_id,
                "audio_url": audio_url,
                "status": "completed",
                "text_length": len(text),
                "processed_text": text,
                "speed": speed,
                "message": "长文本语音合成成功"
            }
        else:
            # 异步处理
            background_tasks.add_task(process_text_task)
            
            # 初始化任务状态
            with open(os.path.join(CACHE_DIR, f"{task_id}.status"), "w") as f:
                f.write("processing")
                
            return {
                "success": True,
                "task_id": task_id,
                "status": "processing",
                "text_length": len(text),
                "processed_text": text,
                "speed": speed,
                "check_status_url": f"{base_url}/status/{task_id}",
                "message": "长文本语音合成任务已开始处理"
            }
    
    except Exception as e:
        logger.error(f"处理长文本请求时出错: {str(e)}")
        # 清理临时文件
        if os.path.exists(prompt_path):
            os.remove(prompt_path)
        raise HTTPException(status_code=500, detail=f"TTS处理错误: {str(e)}")



@app.get("/status/{task_id}")
async def check_task_status(request: Request, task_id: str):
    """检查长文本处理任务的状态"""
    status_path = os.path.join(CACHE_DIR, f"{task_id}.status")
    output_path = os.path.join(OUTPUT_DIR, f"{task_id}.wav")
    
    if not os.path.exists(status_path):
        raise HTTPException(status_code=404, detail=f"任务 {task_id} 不存在")
    
    # 读取任务状态
    with open(status_path, "r") as f:
        status = f.read().strip()
    
    base_url = str(request.base_url).rstrip('/')
    
    if status == "completed" and os.path.exists(output_path):
        return {
            "task_id": task_id,
            "status": status,
            "audio_url": f"{base_url}/audio/{task_id}.wav",
            "completed": True
        }
    elif status.startswith("error"):
        return {
            "task_id": task_id,
            "status": "error",
            "error_message": status[7:] if len(status) > 7 else "未知错误",
            "completed": True
        }
    else:
        return {
            "task_id": task_id,
            "status": status,
            "completed": False
        }



@app.get("/")
async def root():
    """API根路径"""
    return {
        "message": "欢迎使用CosyVoice2 TTS API服务",
        "version": "1.0.0",
        "endpoints": [
            {"path": "/tts", "method": "POST", "description": "标准语音合成"},
            {"path": "/tts_long", "method": "POST", "description": "长文本语音合成，支持异步处理"},
            {"path": "/status/{task_id}", "method": "GET", "description": "检查长文本处理任务的状态"}
        ],
        "status": "ready" if cosyvoice is not None else "initializing"
    }




if __name__ == "__main__":
    import time  # 添加缺少的导入
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)