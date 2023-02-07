# Core Pkgs
import streamlit as st 
import altair as alt
#import plotly.express as px 

# EDA Pkgs
import pandas as pd 
import numpy as np 
from datetime import datetime

# Utils
import joblib 
pipe_lr = joblib.load(open("models/emotion_classifier_pipe_lr_03_june_2021.pkl","rb"))


# Track Utils
from track_utils import add_prediction_details,view_all_prediction_details,create_emotionclf_table

# Fxn
def predict_emotions(docx):
	results = pipe_lr.predict([docx])
	return results[0]

def get_prediction_proba(docx):
	results = pipe_lr.predict_proba([docx])
	return results

emotions_emoji_dict = {"anger":"😠","disgust":"🤮", "fear":"😨😱", "happy":"🤗", "joy":"😂", "neutral":"😐", "sad":"😔", "sadness":"😔", "shame":"😳", "surprise":"😮"}


# Main Application
def main():
	st.title("Emotion Classifier App")
	menu = ["Home","Monitor"]
	choice = st.sidebar.selectbox("Menu",menu)
	# create_page_visited_table()
	create_emotionclf_table()
	if choice == "Home":
		# add_page_visited_details("Home",datetime.now())
		st.subheader("Home-Emotion In Text")

		with st.form(key='emotion_clf_form'):
			raw_text = st.text_area("Type Here")
			submit_text = st.form_submit_button(label='Submit')

		if submit_text:
			#col1,col2  = st.columns(2)

			# Apply Fxn Here
			prediction = predict_emotions(raw_text)
			probability = get_prediction_proba(raw_text)
			
			add_prediction_details(raw_text,prediction,np.max(probability),datetime.now())

			# with col1:
			# 	st.success("Original Text")
			# 	st.write(raw_text)

			# 	st.success("Prediction")
			# 	emoji_icon = emotions_emoji_dict[prediction]
			# 	st.write("{}:{}".format(prediction,emoji_icon))
			# 	st.write("Confidence:{}".format(np.max(probability)))



			# with col2:
			st.success("Prediction Probability")
				# st.write(probability)
			proba_df = pd.DataFrame(probability,columns=pipe_lr.classes_)
				# st.write(proba_df.T)
			proba_df_clean = proba_df.T.reset_index()
			proba_df_clean.columns = ["emotions","probability"]

			fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions',y='probability',color='emotions')
			st.altair_chart(fig,use_container_width=True)

			st.success("suggestions")
			
			if(prediction=="sadness"):
				st.video('https://www.youtube.com/watch?v=hBzP8MtJf04')
				st.video('https://www.youtube.com/watch?v=h-3nt92UFZo')

				st.markdown("##")
				st.markdown("##")

				#st.audio('C:\Users\dell\Downloads\EP 366 Failure Is Your Friend.mp3')
				st.success("motivational podcast")
				audio1=open("audio.mp3","rb")
				st.audio(audio1)
				st.markdown("##")
				st.markdown("##")

				st.success("ACTIVITIES TO MAKE YOU FEEL BETTER")
				st.subheader("1. COMEDY SHOWS")
				st.image("https://th.bing.com/th/id/OIP.Fh-YrCQiNoQKAnV_c1w1QQHaED?w=288&h=180&c=7&r=0&o=5&dpr=1.3&pid=1.7")
				st.write("BOOK YOUR TICKETS BY CLICKING ON THE LINK BELOW")
				st.write("https://in.bookmyshow.com/explore/comedy-shows-bengaluru")

				st.markdown("##")

				st.subheader("2. PLANTATION DRIVE")
				st.image("https://th.bing.com/th/id/OIP.qb39CJ4sHhIHNG0Xk9UJpwAAAA?w=219&h=180&c=7&r=0&o=5&dpr=1.3&pid=1.7")
				
				st.write("REGISTER BY CLICKING ON THE FORM BELOW")
				st.write("https://labouronline.kar.nic.in/Plantation/Plantation_Registration.aspx")

				st.markdown('##')
				st.success("ARTICLES TO OVERCOME SADNESS")
				st.write('https://www.sane.org/information-and-resources/the-sane-blog/my-story/this-is-what-depression-looks-like')


				
				
				

			if(prediction=="sad"):
				st.video('https://www.youtube.com/watch?v=hBzP8MtJf04')
				st.video('https://www.youtube.com/watch?v=h-3nt92UFZo')
			

			if(prediction=="anger"):
				st.video('https://youtu.be/QAsJvKsd2Xk')
				st.video('https://youtu.be/C1N4f1F0vDU')


			if(prediction=="fear"):
				st.video('https://youtu.be/uV_CGpMsEhY')
				st.video('https://youtu.be/TOzJRrGdMCs')
	




	elif choice == "Monitor":
		#add_page_visited_details("Monitor",datetime.now())
		st.subheader("Monitor App")

		# with st.expander("Page Metrics"):
		# 	page_visited_details = pd.DataFrame(view_all_page_visited_details(),columns=['Pagename','Time_of_Visit'])
		# 	st.dataframe(page_visited_details)	

		# 	pg_count = page_visited_details['Pagename'].value_counts().rename_axis('Pagename').reset_index(name='Counts')
		# 	c = alt.Chart(pg_count).mark_bar().encode(x='Pagename',y='Counts',color='Pagename')
		# 	st.altair_chart(c,use_container_width=True)	

		# 	p = px.pie(pg_count,values='Counts',names='Pagename')
		# 	st.plotly_chart(p,use_container_width=True)

		with st.expander('Emotion Classifier Metrics'):
			df_emotions = pd.DataFrame(view_all_prediction_details(),columns=['Rawtext','Prediction','Probability','Time_of_Visit'])
			st.dataframe(df_emotions)

			prediction_count = df_emotions['Prediction'].value_counts().rename_axis('Prediction').reset_index(name='Counts')
			st.write(prediction_count)

			pred=prediction_count['Counts']
				
			#st.write(pred)
			len=pred[0]+pred[1]+pred[2]+pred[3]

			avg=(pred[0]/len)
			st.write(prediction_count["Prediction"][0])
			st.write(avg)
			pc = alt.Chart(prediction_count).mark_bar().encode(x='Prediction',y='Counts',color='Prediction')
			st.altair_chart(pc,use_container_width=True)

			if(prediction_count["Prediction"][0]=="sadness"):
				if(avg>0.6):
					st.write("your average level  is high please contact nearest therapist")
			if(prediction_count["Prediction"][0]=="fear"):
				if(avg>0.6):
					st.write("your average level  is high please contact nearest therapist")




	# else:
	# 	st.subheader("About")
		#add_page_visited_details("About",datetime.now())





if __name__ == '__main__':
	main()