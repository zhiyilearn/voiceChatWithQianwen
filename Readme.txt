配置API Key到环境变量: 
echo "export DASHSCOPE_API_KEY='YOUR_DASHSCOPE_API_KEY'" >> ~/.bashrc
source ~/.bashrc

New window:
echo $DASHSCOPE_API_KEY

conda create -n test1 python=3.11  
conda activate test1

pip install pyaudio
pip install vosk
pip install openai
pip install numpy
pip install pyttsx3 pyaudio webrtcvad

vosk models need unzip in models directory

Execute the code:
in voicelink/interface directory, run:

python3 llmchatwithQianWen_v5.py

