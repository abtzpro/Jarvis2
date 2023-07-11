![Jarvis2 AI Logo](https://github.com/abtzpro/Jarvis2/blob/main/D2468FAD-437B-4A23-B0C3-DA1BF67B27A5.png)

*The real life Ironman JARVIS AI*

# Jarvis2 AI

Welcome to the Jarvis2 AI project! Jarvis2 AI or JarvisAI for short, is an intelligent, multi-modal, and versatile AI platform designed to solve a multitude of tasks ranging from neural network predictions, deep reinforcement learning, web scraping, knowledge graph querying, natural language processing, speech recognition, computer vision, to IoT device control.

## Features

1. **Neural Network Prediction**: TensorFlow-based sequential model for predictions.
2. **Reinforcement Learning**: DQL and PPO based learning using the OpenAI Gym environment.
3. **Web Scraping**: Fetches web content using Requests and BeautifulSoup.
4. **Unsupervised Learning**: KMeans clustering on given datasets.
5. **Knowledge Base**: Neo4j graph database for storing and querying data.
6. **Chatbot**: Chatbot functionality with transformer-based conversational pipeline.
7. **NLP**: Better communication abilities with GPT-3 Language Model and GPT-2 Tokenizer.
8. **Speech Recognition**: Uses the SpeechRecognition library for converting speech to text.
9. **Computer Vision**: Object detection with OpenCV.
10. **Predictive Models**: SVC model from sklearn for predictions.
11. **Emotion AI**: TensorFlow Keras model for emotion recognition.
12. **IoT Device Management**: Home Assistant for managing smart home devices.
13. **Task Scheduling**: Personal assistant functionality for scheduling tasks.
14. **Fact Fetching**: Wolfram Alpha for factual information.

## Prerequisites

The project requires the following dependencies to be installed. Use pip to install them:

```bash
pip install tensorflow
pip install sklearn
pip install transformers
pip install stable-baselines3
pip install neo4j
pip install requests
pip install bs4
pip install gym
pip install SpeechRecognition
pip install opencv-python
pip install python-dotenv
```

## Installation

(Check the releases for the most recent release) 

1. Clone this repository to your local machine.

```bash
git clone https://github.com/abtzpro/Jarvis2
```

2. Navigate to the directory of the project.

```bash
cd Jarvis2
```

3. Install the requirements.

```bash
pip install -r requirements.txt
```

4. Create a `.env` file and store your environment variables there. 

## Usage

(v1.1-Alpha-Build Contains a rudimentary GUI or graphical user interface enabling a user to interact with some of JarvisAI's core functions easier)

Create an instance of the JarvisAI class:

```python
jarvis = JarvisAI(model, agent, driver, chatbot)
```

Now you can use JarvisAI to perform various tasks, such as:

```python
result = jarvis.predict(data)
decision = jarvis.make_decision(observation)
web_content = jarvis.fetch_web_content(url)
query_result = jarvis.query_knowledge_base(query)
response = jarvis.ask_jarvis(question)
```
# Usage (Breakdown) 

The JarvisAI class is essentially a “box” that contains all the features and functionalities provided by the script. These include abilities like making predictions, fetching web content, asking questions to the AI, controlling IoT devices, and more. To use any of these functions, you first need to create an instance of the JarvisAI class.

Creating an instance of a class is a common practice in object-oriented programming. You can think of it as creating a personal copy of the “box” that you can play with. Here is how you do that:
```
jarvis = JarvisAI(model, agent, driver, chatbot)
```
Now, jarvis is an instance (or object) of the JarvisAI class. model, agent, driver, and chatbot are the building blocks of JarvisAI that you’re passing to it. They were all set up in the earlier parts of the script. You don’t need to worry about their inner workings for now.

Once you have your jarvis instance, you can use it to call various methods. Think of these methods as individual tools inside the “box”. For example, if you want JarvisAI to make a prediction based on some data, you can use the predict method:
```
data = [...]  # your data here
prediction = jarvis.predict(data)
```
Similarly, you can use other methods for different tasks. For example, if you want JarvisAI to fetch content from a web page, you can use the fetch_web_content method:
```
url = "http://example.com"  # replace with the URL you want to fetch content from
content = jarvis.fetch_web_content(url)
```
This is the basic idea of how you interact with JarvisAI. If you’re new to programming, some of these concepts might feel a bit complicated, but as you gain more experience, they will become second nature.


## Contributing

We welcome all contributions. You can submit any ideas as pull requests or as GitHub issues. If you'd like to improve code submit a pull request outlining the request.

## License

See the LICENSE file

## Contact

If you want to contact me you can reach me at `https://abtzpro.github.io`.

## Credits

Developed by:

Adam Rivers
(Developer)
https://abtzpro.github.io

Hello Security LLC
(Developer Company)
https://hellosecurityllc.github.io

# Disclaimer: In Active Development

Please note that the Jarvis AI project is currently under active development. This means that the software and its features are subject to change as we continue to make improvements, add new features, and fix bugs. While we strive to maintain a stable and functioning product, please understand that there may be occasional disruptions or inconsistencies in the software's behavior or functionality. 

We appreciate your understanding and patience during this period of active development. We encourage you to report any bugs or issues you encounter, as well as suggest any improvements or features you would like to see in future versions of Jarvis AI. Your feedback is vital for the progress and refinement of this project. Thank you for your ongoing support.

## Special Thanks

This project would not have been possible without the open-source libraries used to bring Jarvis AI to life. A big thank you goes out to the creators and maintainers of TensorFlow, scikit-learn, Beautiful Soup, SpeechRecognition, stable-baselines3, neo4j, gym, and many others. Your work is invaluable for the development of AI and Machine Learning applications. 

Furthermore, a special mention goes to the creators and maintainers of the Transformers library by Hugging Face. The ease-of-use and accessibility of this resource have simplified and accelerated the development of Natural Language Processing tasks and applications. Your efforts are greatly appreciated by this community.

This project is named in homage to Jarvis, the AI assistant of Tony Stark (Iron Man) in the Marvel Cinematic Universe. We want to acknowledge and extend our thanks to the original creators of Iron Man: writer and editor Stan Lee, scripter Larry Lieber, and artists Don Heck and Jack Kirby. Additionally, we want to thank director Jon Favreau and actor Robert Downey Jr. for bringing Iron Man and his iconic AI Jarvis to life on the big screen in such an unforgettable way.

Last but not least, we want to thank all the users and contributors of the Jarvis AI project. Your feedback and contributions drive this project forward and make it better every day. We sincerely appreciate your involvement.
