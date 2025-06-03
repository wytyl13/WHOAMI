#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/04/19 14:24
@Author  : weiyutao
@File    : chattts.py
"""

from typing import (
    Optional,
    Dict
)
import ChatTTS
import asyncio
import numpy as np

from whoami.tool.agent.base_tool import tool
from whoami.tool.agent.tool import TTS
from whoami.tool.agent.tool import AudioType


class ChatParams:
    rand_spk: Optional[str] = None
    params_infer_code: Optional[ChatTTS.Chat.InferCodeParams] = None
    params_refine_text: Optional[ChatTTS.Chat.RefineTextParams] = None
    
    def __init__(
        self, 
        rand_spk: Optional[str] = None, 
        params_infer_code: Optional[Dict] = None, 
        params_refine_text: Optional[Dict] = None
    ):
        self.rand_spk = rand_spk if rand_spk is not None else self.rand_spk
        
        if self.rand_spk is None:
            raise ValueError("rand_spk must not be null")
        
        self.params_infer_code = ChatTTS.Chat.InferCodeParams(
            spk_emb = self.rand_spk, # add sampled speaker 
            temperature = .3,   # using custom temperature
            top_P = 0.7,        # top P decode
            top_K = 20,         # top K decode
        ) if params_infer_code is None else params_infer_code
        
        self.params_refine_text = ChatTTS.Chat.RefineTextParams(
            prompt='[oral_2][laugh_0][break_6]',
        ) if params_refine_text is None else params_refine_text
    




@tool
class ChatTTSImpl(TTS):
    """text to speech used chattts model"""
    audio_type: Optional[AudioType] = None
    chat: Optional[ChatTTS] = None
    chat_params: Optional[ChatParams] = None
    
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if 'audio_type' in kwargs:
            self.audio_type = kwargs.pop('audio_type')
        if 'chat' in kwargs:
            self.chat = kwargs.pop('chat')
        if 'chat_params' in kwargs:
            self.chat_params = kwargs.pop('chat_params')
        
        self.audio_type = AudioType() if self.audio_type is None else self.audio_type
        
        if self.chat is None:
            self.chat = ChatTTS.Chat()
            self.chat.load(compile=False, custom_path='/root/.cache/modelscope/hub/models/AI-ModelScope/ChatTTS')
            
        self.chat_params = ChatParams(
            rand_spk="蘁淰敡欀尤扸尅綐淄朢炊共諴樶瀁婿秀煵茠湎溾縟犐牊萨蔆痳臣瘛暌汌良褓皉箊柁奶滥淨叓侜璐瓕愭眗攞巘蝼湧恅湂癖粯葻虭浜姯摙癄觝佥撟儵渪凊搊怷裡埳觽斻礅磏舯诽夤榋摍繮觎蜟敋怭伉姦稻彪臹讂榠礫絈捥堀僒凼糌嶱膥狊膎慝櫸蛅溪萖瀓惗穞汊撌调柭豏觰棧足盢悷庪罽覉偼凍仒纴篱亝夯箝讝盆橱漳跹愤牷梷襮欀憑縟臐欀綺盅犴蒕澡亊啦菉犰瞋伒洴絒翊剥犰熳糬繞榆賻譩喢杈沙厐砪栦勦嘎他嚍码洓园楣歎導愯绑熹侱薝勶譐牧剭卼憽挻祜娟胁娛禢圼覥俧奥疪嗟瞸犂硌瘋濱蓼崁姫梅诐倔妅矦瑖屙湻珮渮慔廹繃懖紺晓嘐湏沏莉啯撷冚寍筐礣碃此嗨喓蘥暏蜰挕苹媊茤薍婴棙庠滤僔虩葅弐碒脮玞艝帘莨亭剌螙毯倝崐朘洄欪沎赩莏禑啻焨悷廰噇糩埻嵹拶呠掓窸乭汎幟柗伺娑籰牔壧帔嗑哠瑓腟喜僝煌藩嗝巰椡湀笳泸咔荑臜腢彃碻承殓汘竗壥猵椠沜翖杴縗唺窞和槊毐梗繮先痠橺宕蘁昢徝灊檭矲犄殸税廧椷悄倶獠癚猽攟抹穾置膰筈確豊嗝滥谦櫾甆撝蓷徐灗埸蘩晠灝舖狤脋臥伐瀋懕甪媤瘫浹氏廑蔊眊兺繱牏诠睰瓁怭賺衽嚕懘攰憙蠞殽篺藼潠螆繂旈瓷営尚護汲般摜谢懍秄侏濅敗詞抇尩匸敕誎濯煗兢耓烋蚴訟柳趨溾砼賄肤菊舼伛类撨哿奴毂厏怙繝挽娺卷庽徤慹嵖曰僪蜆咈喾极呭圿煥弒厘萹寶搷滟諰翅杫泸榅橠癤臤礲瞪谞識槮瑷測畋濁籊菓攻浡勏儧艰讵俱哤瘦暃涯緭滯檝嵀旇瓰儝泖綅娟奒晢蒒脨娫玍蛻攗譇耡奿樆帎廹莸絩塦怨滛摽虒产忬腵諠牣彸娃讷廥甥祪勐诊弭傤縨淄嵴剕謍瑰侹祳睲忚烮淔攠赇蜁尴嵮旻僡笋谠喰弸曨茕磷褭聞傩磒貘耞萅悽莨濛捍簨赌茕擛汾噖襼蕾坠殆恺蓚洋諮偄褬佲侥佾屛恋拹偓奇秆儬粧榃灄瓰亓贐娄渰湚孓圬弫妫愛滻榤稲榙散犜糾硙箁惤紓覥刖赯搽燴偬紈拘甐翂櫀筶澬螣砡恫摣絁壧傫桂洷箢涖漂茏壗崭娱亅縛叅利幈烖讬啣羴祚箟奢巆絨荾切観攦猨暁歘懬資崜摰瞂独愐率櫅荰匃艍或即弡獐欞寄杇焯裍擿炑幩忈恰厇禗搋説宰巍往妠胑廆泩嵚禛统諕炏敌薎吣伷晧蒊巤湰偓奉汌営膆榤嘿洔垏娡嫦笖偃藷怋怯十溔薚昬蒌凋惰揈嫫起卤佖萠檘蓤卮爇屚抳訶呺豢曟磐舦讠譨睿衧猭報哬裫孺槹莤諴疚甃拎想菐弶衼詰蟱碊耇櫧謚蓊毎怓瀥棒愐耞窯挠挣戃汁吝燪牬覎笉椀纁枒磮潄山擶睉毒爌仏嚼蔲贫皋帽蚋嘗胧癁悫槈宱潍撳俘婪荙簨湞棄樜一㴆",
        ) if self.chat_params is None else self.chat_params
        
        
    async def text_audio(
        self, 
        text=None
    ) -> str:
        try:
            if not text or text.strip() == "":
                self.logger.warning("输入文本为空!")
                # 返回一个小的空音频数组(采样率24000，持续0.1秒的静音)
                return np.zeros(2400, dtype=np.float32)
            results = self.chat.infer(
                text,
                params_refine_text=self.chat_params.params_refine_text,
                params_infer_code=self.chat_params.params_infer_code,
            )
            self.logger.info(text)
            self.logger.info(results)
            if not results:
                self.logger.warning("ChatTTS返回了空结果!")
                return np.zeros(2400, dtype=np.float32)
            
            return results[0]
        except Exception as e:
            raise ValueError(f"Fail to execute the chattts! {str(e)}") from e


if __name__ == '__main__':
    chattts_impl = ChatTTSImpl()
    
    async def main():
        result = await chattts_impl.execute(
            text="越南河内，习近平总书记、国家主席的紧凑行程里，越方的接待彰显高规格。",
            output_path="/work/ai/WHOAMI/222222222.wav",
            audio_type="wav_bytes"
        )
        print(result)
        
    asyncio.run(main())

    