Release v1.1-Alpha-Build > 

-a rough hewn GUI is added.
Access to Jarvis AI features via GUI are documented below.

	1.	Ask Jarvis: The user can input any question into the entry field and then click the “Ask Jarvis” button. Jarvis will attempt to generate a text response to the question.
	2.	Predict: The user can input some data into the entry field and then click the “Predict” button. Jarvis will attempt to make a prediction using the neural network model.
	3.	Make Decision: The user can input an observation into the entry field and then click the “Make Decision” button. Jarvis will make a decision based on the observation using the DQN reinforcement learning agent.
	4.	Make Advanced Decision: Similar to “Make Decision”, but uses the PPO reinforcement learning agent.
	5.	Make Predictive Decision: The user can input some data into the entry field and then click the “Make Predictive Decision” button. Jarvis will attempt to make a prediction using the SVM predictive model.
	6.	Control IoT Device: The user can input a command into the entry field and then click the “Control IoT Device” button. Jarvis will attempt to send this command to the connected Home Assistant instance.
	7.	Schedule Task: The user can input a task into the entry field and then click the “Schedule Task” button. This task will be added to the personal calendar dictionary with the current datetime as the key.
	8.	Fetch Fact: The user can input a query into the entry field and then click the “Fetch Fact” button. Jarvis will attempt to return a factual response using the connected Wolfram Alpha client.
	9.	Generate Text: The user can input a text prompt into the entry field and then click the “Generate Text” button. Jarvis will attempt to generate a piece of text based on this prompt using the GPT-3 model.
	10.	Query Knowledge Base: The user can input a query into the entry field and then click the “Query Knowledge Base” button. Jarvis will attempt to run this query on the connected Neo4j graph database and return the results.

*please note that the effectiveness and flow of these implementations relies solely on whether the models used are trained and will differ depending on your use case. These functions and others can be addressed & assigned in the JarvisAI Class.*

-Some Bugs and errors have been addressed. Streamlining of some core features finished and more comments have been adding for clarity. 
