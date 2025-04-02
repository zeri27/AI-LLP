import os
import re

import edge_tts
import asyncio
import pygame

gen_text = 0

async def tts(text):
    global gen_text
    new_text = re.sub(r'[*\-_]', '', text)
    tts = edge_tts.Communicate(new_text, voice="zh-CN-XiaoxiaoNeural", rate='+15%', volume='-35%', pitch='-30Hz')
    await tts.save(f"output{gen_text}.mp3")

    pygame.mixer.init()
    pygame.mixer.music.load(f"output{gen_text}.mp3")
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        await asyncio.sleep(1)

    gen_text += 1

#text = "I'd be happy to help you practice Mandarin. Since I don't know your current level, let's start with a few simple exercises to get you warmed up. Please choose one from the following: **Exercise 1: Writing Characters** Write out these four Chinese characters: (tī) - to drink, (fēng jī) - windmill, (xiǎo bāo) - small bread, and (wén huì) - culture. **Exercise 2: Sentence Building** Fill in the blank with a suitable verb: _______________________ (wǒ xǐ huan chī fàn). I _______________ eating food. Choose from: è, kà, wèn, or shū."
#asyncio.run(tts(text))