from generator import load_csm_1b, Segment
import torchaudio, torch, os, json
from dataclasses import dataclass

# @dataclass
# class Segment:
#     speaker: int
#     text: str
#     # (num_samples,), sample_rate = 24_000
#     audio: torch.Tensor

@dataclass
class Metrics:
    output_generation_time: float
    average_total_contexts: float
    average_context_duration: float
    output_duration: float

def get_transcripts_from_file():
    results = []
    folder_path = os.path.join(os.getcwd(), "inputs")
        
    for dir_name in os.listdir(folder_path):
        if os.path.isdir(os.path.join(folder_path, dir_name)):
            audio_file = ""
            text = ""

            for file_name in os.listdir(os.path.join(folder_path, dir_name)):
                if file_name.endswith(".txt"):
                    with open(os.path.join(folder_path, dir_name, file_name), "r") as f:
                        text = f.read()
                elif file_name.endswith(".wav"):
                    audio_file = os.path.join(folder_path, dir_name, file_name)

            if len(audio_file) == 0 or len(text) == 0:
                continue
            audio = torchaudio.load(audio_file)[0]
            if audio.shape[0] == 2:  # Stereo to mono
                audio = torch.mean(audio, dim=0, keepdim=True)

            audio = audio.squeeze(0)
            results.append({
                "data": Segment(text=text, speaker=0, audio=audio),
                "name": dir_name
            })
    return results

    

def get_sentences_from_file(file_path, delimiter="\n"):
    result = []
    if not os.path.exists(file_path):
        return result

    with open(file_path, "r") as f:
        result = f.read().split(delimiter)
        result = [r.strip() for r in result if len(r.strip()) > 0]
    return result


    

def get_transcripts_from_text():
    results = []
    text = """
    Colonel Durnford... William Vereker. I hear you 've been seeking Officers?
Good ones, yes, Mr Vereker. Gentlemen who can ride and shoot
Your orders, Mr Vereker?
I'm to take the Sikali with the main column to the river
Lord Chelmsford seems to want me to stay back with my Basutos.
I think Chelmsford wants a good man on the border Why he fears a flanking attack and requires a steady Commander in reserve.
Well I assure you, Sir, I have no desire to create difficulties. 45
And I assure you, you do not In fact I'd be obliged for your best advice. What have your scouts seen?
So far only their scouts. But we have had reports of a small Impi farther north, over there.
Lighting COGHILL' 5 cigar: Our good Colonel Dumford scored quite a coup with the Sikali Horse.
Um. There are rumours that my Lord Chelmsford intends to make Durnford Second in Command.
Well that's typical of Her Majesty's army. Appoint an engineer to do a soldier's work.
Do you think she might be interested in someone?
Which one?
Well that one. The one who keeps looking at me.
ft could be you flatter yourself CoghilL It's that odd eye.
Choose your targets men. That's right Watch those markers. 55
Keep steady. You're the best shots of the TwentyFourth. You bunch of heathens, do it
Stuart?
Yes.
"How quickly can you move your artillery forward?"
    """

    transcripts = [Segment(text=t, speaker=0, audio=None) for t in text.split("\n")]
    return results



def generate_output_for_speech():
    # Load csm related config
    generator = load_csm_1b(device="cuda")

    # Load transcripts
    transcripts = [t["data"] for t in get_transcripts_from_file()]
    join_context_in_final_output = False

    # Config for output and load input text
    output_type = "speech-text"
    sentence_delimiter = "."
    output_dir = os.path.join(os.getcwd(), "outputs", output_type)
    
    if not os.path.exists(output_dir):
        raise ValueError(f"Output directory {output_dir} does not exist")

    # Configure input details
    input_filepath = os.path.join(output_dir, "input.txt")
    if not os.path.exists(input_filepath):
        raise ValueError(f"Input file {input_filepath} does not exist") 

    input_texts = get_sentences_from_file(input_filepath, delimiter=sentence_delimiter)
    output_filename = "audio.wav"
    output_filepath = os.path.join(output_dir, output_filename)

    # add transcripts as context
    context = []
    for i in range(len(transcripts)):
        if transcripts[i].audio is None:
            audio = generator.generate(
                text=transcripts[i].text,
                speaker=transcripts[i].speaker,
                context=context,
                max_audio_length_ms=10_000,
            )
            transcripts[i].audio = audio
        context.append(transcripts[i])
        print(f"Generated audio for segment {i}. Remaining segments: {len(transcripts) - i - 1}")

    # Generate output
    output_audios = []
    for i in range(len(input_texts)):
        input_text = input_texts[i]
        output = generator.generate(
            text=input_text,
            speaker=0,
            context=context,
            max_audio_length_ms=10_000,
        )  
        output_audios.append(output.cpu())
        print(f"Generated audio for input text. Remaining input texts: {len(input_texts) - i - 1}")

    output = torch.cat(output_audios, dim=0).unsqueeze(0)
    print(f"Generated audio for input text. Saving to {output_filepath}")
    torchaudio.save(output_filepath, output, generator.sample_rate)


def generate_output_for_conversation():
    # Load csm related config
    generator = load_csm_1b(device="cuda")
    # Load transcripts
    transcripts = get_transcripts_from_file()
    join_context_in_final_output = False

    # Config for output and load input text
    output_type = "conversation-text"
    sentence_delimiter = "."
    output_dir = os.path.join(os.getcwd(), "outputs", output_type)
    
    if not os.path.exists(output_dir):
        raise ValueError(f"Output directory {output_dir} does not exist")


    # Configure input details
    input_filepath = os.path.join(output_dir, "input.json")
    if not os.path.exists(input_filepath):
        raise ValueError(f"Input file {input_filepath} does not exist") 

    input_texts = []
    with open(input_filepath, "r") as f:
        texts = json.load(f)
        for t in texts:
            # split text by seperator
            sentences = t["value"].split(sentence_delimiter)
            for s in sentences:
                input_texts.append({
                    "name": t["name"],
                    "data": s
                })
    print(input_texts)

    output_filename = "audio.wav"
    output_filepath = os.path.join(output_dir, output_filename)

    # Generate output
    output_audios = []
    for i in range(len(input_texts)):
        input_text = input_texts[i]
        print(f"Generating audio for input text: {input_text['data']} by speaker {input_text['name']}")
        extracted_transcripts = []
        for t in transcripts:
            if int(t["name"]) >= 5:
                if input_text["name"] == "Developer":
                    extracted_transcripts.append(t["data"])
            else:
                if input_text["name"] == "Project Manager":
                    extracted_transcripts.append(t["data"])
        print(f"Extracted transcripts: {extracted_transcripts}")

        output = generator.generate(
            text=input_text['data'],
            speaker=0,
            context=extracted_transcripts,
            max_audio_length_ms=10_000,
        )  
        output_audios.append(output.cpu())
        print(f"Generated audio for input text. Remaining input texts: {len(input_texts) - i - 1}")

    output = torch.cat(output_audios, dim=0).unsqueeze(0)
    print(f"Generated audio for input text. Saving to {output_filepath}")
    torchaudio.save(output_filepath, output, generator.sample_rate)


   

if __name__ == "__main__":
    generate_output_for_conversation()