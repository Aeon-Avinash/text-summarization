# Load model directly without pipeline
from transformers import pipeline
import torch
import gradio as gr
from litserve import LitAPI, LitServer


model_name="sshleifer/distilbart-cnn-12-6"
text_summary_pipeline = pipeline("summarization", model=model_name, device=-1, torch_dtype=torch.bfloat16)          
# use device=-1 for CPU and device=0 for GPU (using litserve we could automate it, but without using a wrapper class 'device' has to be manually specified) 

text = '''
As soon as a woman becomes pregnant, their body begins to prepare for safeguarding and maintaining the pregnancy. This increases levels of the hormones oestrogen and progesterone in their blood. Read more about what these hormones do in your body in our article about pregnancy hormones.

Higher levels of progesterone and oestrogen are important for a healthy pregnancy, but are often the cause of some common unwanted side effects. This is especially true in the first trimester.

Apart from sickness and tiredness, it's common to have mood swings and feel tearful or easily irritated. Once the body has adapted to the higher levels of these hormones, the symptoms usually wear off. However, some women will experience them throughout their pregnancy. 

Aside from emotional ups and downs caused by rising hormone levels in the first three months, the feeling of growing a new life can be exciting and awe-inspiring. It is also common to feel anxious, vulnerable and overwhelmed by the big changes that pregnancy and a new baby will bring. This can be particularly true for parents who are pregnant after previous loss or following fertility treatment.

Even if you feel excited by the pregnancy, you may have some unsettling thoughts. Perhaps there will also be some difficult decisions to make. Many women have questions that they ask of themselves. They might doubt their ability as a mother, how their relationship might change or how they will manage financially. Other normal worries include:

It can be hard to think clearly or feel positive when you are feeling worried and tired. Taking good physical care of yourself, especially getting plenty of rest and sleep, may help to keep troubling emotions in proportion.

Gentle to moderate exercise can help to improve mood and general fitness in pregnancy, helping you prepare for labour and avoid some complications of pregnancy. Try to build in some activity every day. Avoid contact sports or any strenuous exercise, particularly if you weren’t active before your pregnancy.

Finding out about benefit entitlements, midwife appointments, how you can eat healthily in pregnancy and what you might prepare for your baby can feel overwhelming. So having a to-do list can help you get these things organised in your mind. This NHS to-do list contains lots of useful information. Maybe share your to-do list with your partner or a supportive friend or relative; they might be able to offer you support in ticking some items off that list.  

Bottling up concerns could increase your anxiety. Discussing your feelings and worries with someone who makes you feel comfortable can help you regulate your emotions and limit worry and anxiety.

Talking to other expectant parents may also reveal that you are not alone in your experiences, as well as providing peer support. Joining an NCT antenatal course or a ‘bumps and babies’ group can give you an instant support network. You can find out here what local NCT activities are happening in your area.

It may help to give yourself a rest, focus on your unborn baby and take time to enjoy the pregnancy. Or it might help to spend some time thinking about and doing things that aren’t related to the pregnancy. Maybe that includes indulging in your favourite hobby, catching up with friends or watching the new box office hit at the cinema.

Practising mindfulness techniques can be another useful way of managing big or changeable emotions. Using mindfulness could help you stay in the present moment, and provide you with other skills to help you deal with stressful situations and anxieties in pregnancy.
'''

# print(text_summary_pipeline(text))

def summary (input):
    output = text_summary_pipeline(input)
    return output[0]['summary_text']

gr.close_all();

# version 0.1
# demo = gr.Interface(fn=summary, inputs="text", outputs="text")
# version 0.2
demo = gr.Interface(
    fn=summary, 
    inputs=[gr.Textbox(label="Input text to summarize", lines=6)], 
    outputs=[gr.Textbox(label="Text Summary", lines=4)], 
    title="GenAI Text Summarizer", 
    description="This app summarizes text input")
demo.launch();
