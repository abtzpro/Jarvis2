# Jarvis2 AI
The real life Ironman JARVIS AI

## JARVIS2 AI Project

The JARVIS2 AI Project is an ambitious endeavor by Adam Rivers to replicate a real-life JARVIS (Just A Rather Very Intelligent System), akin to that seen in the Marvel Cinematic Universe in the Iron Man movies. Our JARVIS is designed with capabilities in diverse areas, including machine learning, deep learning, web scraping, natural language processing (NLP), computer vision, reinforcement learning, and much more.

## Features

- **Deep Learning Models**: JARVIS utilizes a Sequential model from TensorFlow for varied tasks like classification, regression etc.

- **Reinforcement Learning**: JARVIS employs both the DQN and the PPO algorithms from stable_baselines3, interacting with environments from OpenAI's gym.

- **Web Content Fetching**: The AI can fetch and parse web content using `requests` and `BeautifulSoup`.

- **Unsupervised Learning**: For clustering needs, JARVIS has KMeans algorithm implementation.

- **Knowledge Base**: Neo4j graph database is leveraged for maintaining a robust and scalable knowledge base.

- **Conversational Abilities**: Enabled by HuggingFace's Transformers library, JARVIS possesses advanced conversational capabilities.

- **Speech Recognition**: The AI can recognize and understand spoken language using the `speech_recognition` Python library.

- **Computer Vision**: JARVIS 'sees' and processes visual data with a pre-trained model in OpenCV.

- **Predictive Models**: JARVIS uses a Support Vector Machine (SVM) model for making predictive decisions.

- **Emotion Recognition**: An emotion AI model is included that helps JARVIS understand human emotions from audio inputs.

- **IoT Integration**: Interfacing with the Home Assistant platform allows JARVIS to control IoT devices.

- **Personal Assistant Tasks and Scheduling**: The AI can manage tasks and schedules.

## Usage

The primary class is `JarvisAI`. After creating an instance of this class, users can access the various functionalities.

```python
# Create a JarvisAI object
jarvis = JarvisAI(model, agent, driver, chatbot)

# Get a prediction from the model
prediction = jarvis.predict(data)

# Fetch web content
soup = jarvis.fetch_web_content("http://example.com")
```

## Installation

To run the JARVIS AI project, you'll need to install several libraries and frameworks, including TensorFlow, sklearn, transformers, stable_baselines3, neo4j, BeautifulSoup, gym, speech_recognition, cv2, wikipedia, and wolframalpha. 

Ensure all the file paths, URLs, and credentials (for example, Wolfram Alpha App ID, Neo4j authentication) in the script are correctly set up.

## Active-Dev Disclaimer 
Please note that the JARVIS AI Project is still in the active development stage. As such, it is likely to contain bugs, glitches, or other issues. These may affect the project’s stability and performance. We appreciate your patience and understanding as we work diligently to iron (man) out these issues and improve the system. Please feel free to report any bugs or problems you encounter, and we’ll do our best to address them promptly. We welcome all feedback and suggestions that can help enhance the development and effectiveness of this project. Thank you for your support!

## Credits
*Developed By*

Adam Rivers 
(Developer)
https://abtzpro.github.io

Hello Security LLC 
(Developer Company)
https://hellosecurityllc.github.io

*Extra Credit to:*

Marvel Studios for the original Iron Man JARVIS AI insipiration.

OpenAI For the Gym functions and access given via API and huggingface of the gpt2 pretrained model. 

Huggingface for the ease of access and hosting of important AI models. 

## Contributions

Your contributions, issues, and feature requests are welcome. Feel free to check the issues page if you want to contribute.

## License

(See LICENSE file)
