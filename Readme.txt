配置API Key到环境变量: 
echo "export DASHSCOPE_API_KEY='YOUR_DASHSCOPE_API_KEY'" >> ~/.bashrc
source ~/.bashrc

New window:
echo $DASHSCOPE_API_KEY

conda create -n test1 python=3.11  
conda activate test1

pip install vosk
pip install openai
pip install numpy
pip install pyttsx3 pyaudio webrtcvad
pip install scipy
pip install noisereduce
pip install openai-whisper -i https://pypi.tuna.tsinghua.edu.cn/simple


pip install -U openai-whisper

vosk models need unzip in models directory

Execute the code:
in voicelink/interface directory, run:

python3 llmchatwithQianWen.py



