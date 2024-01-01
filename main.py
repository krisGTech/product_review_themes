import streamlit as st 
from streamlit_extras.metric_cards import style_metric_cards
import plotly_express as px
import plotly 
from bertopic import BERTopic
from textblob import TextBlob
import plotly.graph_objects as go
from plotly.graph_objects import Layout
import numpy as np 
import time 
# Import Model 
review_model = BERTopic.load('product_reviews_tp_model')

st.title("Customer Product Review Themes Prediction") 


col5, col6 = st.columns(2)

    # file_ = open('./image/hospital.gif','rb')

    # contents = file_.read()

    # data_url = base64.b64encode(contents).decode('utf-8')

    # file_.close()

    # st.markdown(f'<image src="data:image/gif;base64,width="250" height="250",{data_url}" alt="hospital gif">',unsafe_allow_html=True)

with col5:
    st.image('review_png.png',width=370)

with col6:

    st.write("""Revolutionizing Insights with Our Customer Review Topic Generation Web App.
At D3.ai, innovation meets insight, and we are thrilled to unveil our latest tool 
             designed to transform the way businesses understand and leverage customer 
             feedback – the Customer Review Topic Generation Web App. In a world where customer 
             opinions shape brand success, our app empowers organizations to dive deep into 
             the intricacies of customer reviews, unveiling valuable insights that 
             drive strategic decision-making. """)





text = st.text_area("Please Enter Review Text Here ...")


def create_probability_graph_plotly(digits, probabilities,text):
    # Check if the input lists have the correct length
    if len(digits) != 5 or len(probabilities) != 5:
        raise ValueError("Both lists must have exactly 5 elements.")

    # Convert numeric digits to object data type
    digits_str = [str(digit) for digit in digits]

    # Define custom x-axis labels using a dictionary
    custom_labels =     {-1: 'top not but so',
                         0: 'vest look not well',
                         1: 'romper short leg well',
                         2: 'tunic wear legging very',
                         3: 'cardigan fall will but',
                         4: 'jacket coat not fit',
                         5: 'sweater not but very',
                         6: 'cami underneath tank black',
                         7: 'tank love great top',
                         8: 'skirt great size fit',
                         9: 'blouse look not size',
                         10: 'shirt not but so',
                         11: 'fabric color not back',
                         12: 'short these length leg',
                         13: 'jean these fit waist',
                         14: 'pant these pair not',
                         15: 'great fit price love',
                         16: 'suit cup swimsuit medium',
                         17: 'bra but not top',
                         18: 'chest small cheste want',
                         19: 'top size but not',
                         20: 'top love fabric not',
                         21: 'tee so basic love',
                         22: 'dress but not fit'}

    # Map the numeric digits to custom labels
    x_axis_labels = [custom_labels[digit] for digit in digits]

    # Create a bar graph using plotly
    layout = Layout(plot_bgcolor='rgba(0,0,0,0)')
    # Use that layout here
    #fig = go.Figure()
    
    fig = go.Figure(layout=layout)
    # Find the index of the max value
    #max_index = probabilities.index(max(probabilities))
    max_index = np.min(probabilities)

    # Create a list of colors based on whether it's the max value
    #colors = ['purple' if i == max_index else 'gray' for i in range(len(probabilities))]
    colors = 'orange'

    fig.add_trace(go.Bar(x=probabilities , y=x_axis_labels, 
                         #marker_color='purple',
                         marker_color=colors,  # Customize bar color
                         orientation='h',
                         text=probabilities,  # Display values on top of bars
                        textposition='outside',  # Position text outside the bars
                       
                          
                        ))

    # Set labels and title
    fig.update_layout(
        xaxis_title='Probabilities Score',
        yaxis_title='Themes',
        yaxis = dict(tickfont=dict(size=18)),
       # title='Probability Distribution for Topic',
          # Center the title
        title_font=dict(color='gray'),
        title=dict(text="Probability Distribution For Themes Reviews", font=dict(size=18), automargin=True, yref='paper'),
        title_x=0.5,
        



    )
    fig.update_layout(yaxis=dict(autorange="reversed"))

    # Show the graph
    st.plotly_chart(fig)
    #print('Sentifment Score',textblob.TextBlob(text).sentiment.polarity)






if st.button("Predict Review Text Themes"): 
  with st.spinner('One moment please featching you the predited theme ...'):
        time.sleep(5)
  #st.spinner(text="Generating Topics...")
  num_of_topics = 5 
  sim_topic,similarity = review_model.find_topics(text,top_n=num_of_topics);
  similarity_ = np.round(similarity,2)
  blob = TextBlob(text) 
  result = blob.sentiment.polarity
  result = np.round(result,3)
  #col1 = st.columns(1)
  #col1.metric(result,label='Sentiment')
  #st.write('Text Sentiment Score:', result)
  st.write(f" <style>...</style> Text Sentiment Score: {result}",unsafe_allow_html=True)

  #st.success(result)
  #st.markdown('score',result)
  #st.write(sim_topic,similarity)
  create_probability_graph_plotly(sim_topic, similarity_,text)
