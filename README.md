**Overview**: What determines wholesale electricity price spikes? I will obtain wholesale electricity pricing data and combine that with temperature, date, fuel price, and other weather data. I will then try to predict price spikes and the magnitude of price spikes, and compare them with actual data, to see if they are indeed predictable and what the strongest predictors are. 

**Session 5 Checkpoint:**
- Data Collection and Cleaning: Pull historical wholesale electricity pricing data from ERCOT or PJM (or both)'s public data portals. At the same time, gather corresponding weather data for the same region, date, time span. Furthermore, gather available fuel price data, for instance natural gas spot prices from EIA. It might also be useful to gather calendar metadata, for instance weekends, holidays. Merge everything into a single dataframe, indexed by date and timestamp. 
- Initial Data Analysis: Compute summary statistics for electricity prices. Use these summary statistics to define what constitutes a price "spike", and if there is a need to categorize these spikes according to how severe they are. Visualize or elaborate on any seasonal trends and time of day patterns, as well as any relationships between weather and price. 
- Problem Scoping: Define the prediction task I am trying to complete here. Also, lay out how I will accomplish this task: which models am I training? Which data am I using (e.g. wind speed, temperature, cloud cover, natural gas price)?


**Session 7 Checkpoint:**
- Simpler Models: Pick the simplest model and a simple question from the scope laid out in Session 5, and try to answer those questions. Come up with initial results, i.e. how successful are these simple models at predicting spikes? 
- Testing Framework: How will I test these models? Will I compare them against one another, to see which situations they perform better in? Do the models contradict one another? Does regularization improve results? 
- Advanced Models: Train 1 or 2 advanced mdoels, such as a KNN classifier and/or random forest. Compare their results using the framework developed. 
- Tune Parameters: Are there structural changes in the grid that result in models being more accurate with more recent data? 

**Final Project Checkpiont:**
- Spike or no Spike: Articulating Predictors -- Which predictors are strongest? Is there more than one "type" of spike? 
- Magnitude of Spikes: Are these predictable to the same degree as the presence/absence of a spike? 
- Model Selection: which model performed the best and why? 
- Film video

**Syllabus Requirements**
What’s creditworthy for the final project? In each of the content and delivery categories respectively, it’s this:

Content: 
Project includes sufficient scope of work (28 points total over the quarter)
You’ll get feedback on scope when you submit your proposal.
It’s a 7-week project, not a doctoral thesis. We will keep this in mind. Any type of “study” you undertake, we’ll expect a “preliminary exploratory study” level of complexity and rigor, not an “academic paper” level of rigor.
Target explainability and configurability in your projects. Clearly, I’m asking you to do data automation projects, and LLMs are very sexy right now. But, in the example above, I would not approve a project that’s like “my recommender system is to catapult the consumer’s purchase log into an LLM and show them what it said,” because you cannot explain how those recommendations were generated. Though prompt engineering is a useful skill, what we’re doing here is learning to actually design data algorithms, not send a long list of markdown instructions over HTTP to a foundation model, cross your fingers, and hope it complies. Please keep that in mind.
Delivery: 
Artifacts that convince me you’ve done the portion of the project you told me you’d do (14 points total over the quarter).
Clearly, with projects being so open-ended, these could be a lot of things. For the project proposals above in order this might be: 
A code base including your different tries at the “Random Forest LLM,” with a summary of the results you got upon running each one
The illustrated board book—ideally what it looked like at a few different stages of completion.
Bert’s bike lane data, the analysis code, the visualizations, and the conclusions.
