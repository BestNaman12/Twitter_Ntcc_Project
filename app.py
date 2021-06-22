import streamlit as st
import tweepy
from textblob import TextBlob
from wordcloud import WordCloud
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
plt.style.use('fivethirtyeight')

consumerKey = 'dafJg8Hqgh4desUtcHmeX5ZvL'
consumerSecret = '0P7AV8uZIdpBmN6XdeeJtz2i3n2bwiAI5Hyx8CCS5HW7ZSfzmi'
accessToken = '1368206717457068035-kZufQ96RxtKQsEZLCD06DBw6nhFQzQ'
accessTokenSecret = '4JkHpc1y2dLA1fDeaNgMlEREigW6EsIXLByCxLoQT3awP'

try:
    # Create the authentication object 
    authenticate = tweepy.OAuthHandler(consumerKey, consumerSecret) 
    # Set the access token and access token secret
    authenticate.set_access_token(accessToken, accessTokenSecret) 
    # Creating the API object while passing in auth information
    api = tweepy.API(authenticate, wait_on_rate_limit = True) 
except:
    print("Invalid Credentials")

def app():
	st.title("🔥 Twitter Sentiment Analyzer 🔥")
	activities=["Tweets Sentiment Analyzer","Generate Twitter Data"]
	choice = st.sidebar.selectbox("Select Your Required Option : ",activities)
	
	if choice=="Tweets Sentiment Analyzer":
		st.subheader("You can analyse the sentiment of tweets of anyone who is using twitter ")
		st.subheader("This tool performs the following tasks :")
		st.write("1. Fetches the 100 most recent tweets from the given twitter handle")
		st.write("2. Generates a Word Cloud")
		st.write("3. Show Positive Tweets Only ")
		st.write("4. Show Negative Tweets Only")
		st.write("5. Show Percentage Distribution of Tweets ")
		st.write("6. Performs Sentiment Analysis a displays it in form of a Bar Graph")

		raw_text = st.text_area("Enter the exact twitter handle (without @) : ")
		Analyzer_choice = st.selectbox("Select the Option from Dropdown ",["Show Recent Tweets","Generate WordCloud" ,"Show Positive tweets","Show Negative tweets","Show Percentage Distribution of Tweets","Generate the bar graph of Sentiment Analysis"])

		if st.button("Generate Data "):
			if Analyzer_choice=="Show Recent Tweets":
				st.success("Fetching last 100 Tweets")
	
				def Show_Recent_Tweets(raw_text):
					# Extract 100 tweets from the twitter user
					posts = api.user_timeline(screen_name=raw_text, count = 100, lang ="en", tweet_mode="extended")
				
					def get_tweets():

						l=[]
						i=1
						for tweet in posts[:100]:
							l.append(tweet.full_text)
							i= i+1
						return l

					recent_tweets=get_tweets()		
					return recent_tweets

				recent_tweets= Show_Recent_Tweets(raw_text)

				st.write(recent_tweets)

			elif Analyzer_choice=="Generate WordCloud":
				st.success("Generating Word Cloud")
				def gen_wordcloud():
					posts = api.user_timeline(screen_name=raw_text, count = 100, lang ="en", tweet_mode="extended")


					# Create a dataframe with a column called Tweets
					df = pd.DataFrame([tweet.full_text for tweet in posts], columns=['Tweets'])
					# word cloud visualization
					allWords = ' '.join([twts for twts in df['Tweets']])
					wordCloud = WordCloud(width=500, height=300, random_state=21, max_font_size=110).generate(allWords)
					plt.imshow(wordCloud, interpolation="bilinear")
					plt.axis('off')
					plt.savefig('WordCloud.jpg')
					img= Image.open("WordCloud.jpg") 
					return img

				img=gen_wordcloud()

				st.image(img)
			elif Analyzer_choice=="Show Positive tweets":
				st.success("Fetching Positive Tweets : ")
				l1 = []
				def Positive_tweets():
					posts = api.user_timeline(screen_name=raw_text, count = 100, lang ="en", tweet_mode="extended")
					# Create a dataframe with a column called Tweets
					df = pd.DataFrame([tweet.full_text for tweet in posts], columns=['Tweets'])
					def cleanTxt(text):
					 text = re.sub('@[A-Za-z0–9]+', '', text) #Removing @mentions
					 text = re.sub('#', '', text) # Removing '#' hash tag
					 text = re.sub('RT[\s]+', '', text) # Removing RT
					 text = re.sub('https?:\/\/\S+', '', text) # Removing hyperlink
					 
					 return text


					# Clean the tweets
					df['Tweets'] = df['Tweets'].apply(cleanTxt)


					def getSubjectivity(text):
					   return TextBlob(text).sentiment.subjectivity

					# Create a function to get the polarity
					def getPolarity(text):
					   return  TextBlob(text).sentiment.polarity


					# Create two new columns 'Subjectivity' & 'Polarity'
					df['Subjectivity'] = df['Tweets'].apply(getSubjectivity)
					df['Polarity'] = df['Tweets'].apply(getPolarity)


					def getAnalysis(score):
					  if score < 0:
					    return 'Negative'
					  elif score == 0:
					    return 'Neutral'
					  else:
					    return 'Positive'
					    
					df['Analysis'] = df['Polarity'].apply(getAnalysis)
					j=0
					sortedDF = df.sort_values(by=['Polarity']) #Sort the tweets
					for i in range(0, sortedDF.shape[0] ):
  						if(sortedDF['Analysis'][i] == 'Positive'):
							  j=j+1
							  l1.append((sortedDF['Tweets'][i]))
					return l1
				positive_tweets_collector = Positive_tweets()
				st.write(positive_tweets_collector)
			
			elif Analyzer_choice==  "Show Negative tweets":
				st.success("Fetching Negative Tweets : ")
				l2 = []
				def Negative_tweets():
					posts = api.user_timeline(screen_name=raw_text, count = 100, lang ="en", tweet_mode="extended")
					# Create a dataframe with a column called Tweets
					df = pd.DataFrame([tweet.full_text for tweet in posts], columns=['Tweets'])
					def cleanTxt(text):
					 text = re.sub('@[A-Za-z0–9]+', '', text) #Removing @mentions
					 text = re.sub('#', '', text) # Removing '#' hash tag
					 text = re.sub('RT[\s]+', '', text) # Removing RT
					 text = re.sub('https?:\/\/\S+', '', text) # Removing hyperlink
					 
					 return text


					# Clean the tweets
					df['Tweets'] = df['Tweets'].apply(cleanTxt)


					def getSubjectivity(text):
					   return TextBlob(text).sentiment.subjectivity

					# Create a function to get the polarity
					def getPolarity(text):
					   return  TextBlob(text).sentiment.polarity


					# Create two new columns 'Subjectivity' & 'Polarity'
					df['Subjectivity'] = df['Tweets'].apply(getSubjectivity)
					df['Polarity'] = df['Tweets'].apply(getPolarity)


					def getAnalysis(score):
					  if score < 0:
					    return 'Negative'
					  elif score == 0:
					    return 'Neutral'
					  else:
					    return 'Positive'
					    
					df['Analysis'] = df['Polarity'].apply(getAnalysis)
					sortedDF = df.sort_values(by=['Polarity'],ascending=False) #Sort the tweets
					for i in range(0, sortedDF.shape[0] ):
  						if(sortedDF['Analysis'][i] == 'Negative'):
							  l2.append((sortedDF['Tweets'][i]))
					return l2
				negative_tweets_collector = Negative_tweets()
				st.write(negative_tweets_collector)
			elif Analyzer_choice==  "Show Percentage Distribution of Tweets":
				posts = api.user_timeline(screen_name=raw_text, count = 100, lang ="en", tweet_mode="extended")
					# Create a dataframe with a column called Tweets
				df = pd.DataFrame([tweet.full_text for tweet in posts], columns=['Tweets'])
				def cleanTxt(text):
				 text = re.sub('@[A-Za-z0–9]+', '', text) #Removing @mentions
				 text = re.sub('#', '', text) # Removing '#' hash tag
				 text = re.sub('RT[\s]+', '', text) # Removing RT
				 text = re.sub('https?:\/\/\S+', '', text) # Removing hyperlink
				 
				 return text

				# Clean the tweets
				df['Tweets'] = df['Tweets'].apply(cleanTxt)

				def getSubjectivity(text):
				   return TextBlob(text).sentiment.subjectivity
					# Create a function to get the polarity
				def getPolarity(text):
				   return  TextBlob(text).sentiment.polarity

				# Create two new columns 'Subjectivity' & 'Polarity'
				df['Subjectivity'] = df['Tweets'].apply(getSubjectivity)
				df['Polarity'] = df['Tweets'].apply(getPolarity)

				def getAnalysis(score):
				  if score < 0:
				    return 'Negative'
				  elif score == 0:
				    return 'Neutral'
				  else:
				    return 'Positive'
				    
				df['Analysis'] = df['Polarity'].apply(getAnalysis)
				ptweets = df[df.Analysis == 'Positive']
				ptweets = ptweets['Tweets']
				posit=round( (ptweets.shape[0] / df.shape[0]) * 100 , 1)
				st.write("Positive Tweets percentage : ",posit)
				ntweets = df[df.Analysis == 'Negative']
				ntweets = ntweets['Tweets']
				negat=round( (ntweets.shape[0] / df.shape[0]) * 100 , 1)
				st.write("Negative Tweets percentage : ",negat)
				neutral_perc = 100 - (posit+negat)
				st.write("Neutral Tweets Percentage : ",neutral_perc)
				st.write("\n")
				st.write("Value Counts of Tweets : \n ")
				st.write(df['Analysis'].value_counts())
			else:



				
				def Plot_Analysis():

					st.success("Generating Bar Graph for Sentiment Analysis")

					st.set_option('deprecation.showPyplotGlobalUse', False)



					posts = api.user_timeline(screen_name=raw_text, count = 100, lang ="en", tweet_mode="extended")

					df = pd.DataFrame([tweet.full_text for tweet in posts], columns=['Tweets'])


					
					# Create a function to clean the tweets
					def cleanTxt(text):
					 text = re.sub('@[A-Za-z0–9]+', '', text) #Removing @mentions
					 text = re.sub('#', '', text) # Removing '#' hash tag
					 text = re.sub('RT[\s]+', '', text) # Removing RT
					 text = re.sub('https?:\/\/\S+', '', text) # Removing hyperlink
					 
					 return text


					# Clean the tweets
					df['Tweets'] = df['Tweets'].apply(cleanTxt)


					def getSubjectivity(text):
					   return TextBlob(text).sentiment.subjectivity

					# Create a function to get the polarity
					def getPolarity(text):
					   return  TextBlob(text).sentiment.polarity


					# Create two new columns 'Subjectivity' & 'Polarity'
					df['Subjectivity'] = df['Tweets'].apply(getSubjectivity)
					df['Polarity'] = df['Tweets'].apply(getPolarity)


					def getAnalysis(score):
					  if score < 0:
					    return 'Negative'
					  elif score == 0:
					    return 'Neutral'
					  else:
					    return 'Positive'
					    
					df['Analysis'] = df['Polarity'].apply(getAnalysis)


					return df



				df= Plot_Analysis()



				st.write(sns.countplot(x=df["Analysis"],data=df))


				st.pyplot(use_container_width=True)

				

	

	else:

		st.subheader("This tool fetches the last 100 tweets from the twitter handel & Performs the following tasks")

		st.write("1. Converts it into a DataFrame")
		st.write("2. Cleans the text")
		st.write("3. Analyzes Subjectivity of tweets and adds an additional column for it")
		st.write("4. Analyzes Polarity of tweets and adds an additional column for it")
		st.write("5. Analyzes Sentiments of tweets and adds an additional column for it")






		user_name = st.text_area("*Enter the exact twitter handle of the Personality (without @)*")

		st.markdown("<--------     Also Do checkout the another cool tool from the sidebar")

		def get_data(user_name):

			posts = api.user_timeline(screen_name=user_name, count = 100, lang ="en", tweet_mode="extended")

			df = pd.DataFrame([tweet.full_text for tweet in posts], columns=['Tweets'])

			def cleanTxt(text):
				text = re.sub('@[A-Za-z0–9]+', '', text) #Removing @mentions
				text = re.sub('#', '', text) # Removing '#' hash tag
				text = re.sub('RT[\s]+', '', text) # Removing RT
				text = re.sub('https?:\/\/\S+', '', text) # Removing hyperlink
				return text

			# Clean the tweets
			df['Tweets'] = df['Tweets'].apply(cleanTxt)


			def getSubjectivity(text):
				return TextBlob(text).sentiment.subjectivity

						# Create a function to get the polarity
			def getPolarity(text):
				return  TextBlob(text).sentiment.polarity


						# Create two new columns 'Subjectivity' & 'Polarity'
			df['Subjectivity'] = df['Tweets'].apply(getSubjectivity)
			df['Polarity'] = df['Tweets'].apply(getPolarity)

			def getAnalysis(score):
				if score < 0:
					return 'Negative'

				elif score == 0:
					return 'Neutral'


				else:
					return 'Positive'

		
						    
			df['Analysis'] = df['Polarity'].apply(getAnalysis)
			return df

		if st.button("Show Data"):

			st.success("Fetching Last 100 Tweets")

			df=get_data(user_name)

			st.write(df)






	st.subheader(':sunglasses: CREATED BY NAMAN MITTAL AND PRANJUL GARG :sunglasses:')

if __name__ == "__main__":
	app()