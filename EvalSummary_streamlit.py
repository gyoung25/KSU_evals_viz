import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from matplotlib.colors import ListedColormap
from eval_summary_helpers import *
import warnings
warnings.filterwarnings('ignore')

###############################################
###### Make widget for Excel file upload ######
###############################################

st.set_page_config(layout="wide")

st.title('KSU Student Evaluation Visualization')

st.write('''#### **Upload raw data files**:
- Download the raw data (Excel) file(s) containing your student evaluation data from Watermark.
    - Note: This app will only work for course evaluation from Fall 2023 onward. The current survey was introduced that semester.
- Drag and drop (or otherwise upload) the raw data files to the field labeled "Drag and drop files here"
    - The raw data files can have any name, but don't change anything within the files themselves before uploading.
    - You can upload files one at a time or all together.
    - The app will ignore any duplicate files, so you don't need to worry if you upload the same file twice.
''')
uploaded_files = st.file_uploader(
    "Please upload raw data xlsx file(s).", accept_multiple_files=True
)

questions = {'Question 1': 'The instructor was effective in helping me learn.',
    'Question 2': 'The instructor created a learning environment wherein I felt comfortable participating.',
    'Question 3': 'Overall, the content of this course contributed to my knowledge and skills.',
    'Question 4': 'Please provide your feedback on the instructor\'s role in supporting your learning in this course.',
    'Question 5': 'Please comment on the instructor\'s strengths.',
    'Question 6': 'Please comment on how the instructor can improve your learning in this class.',
    'Question 7': 'Please comment on how the course can be improved.'}

question_list = ['Question 1', 'Question 2', 'Question 3']
###########################
###### Prepare data #######
###########################


if len(uploaded_files)>0:
    question_data = concat_dfs(uploaded_files)
    question_data = sort_survey_data(question_data)

    st.write('#### **Visualize student responses**')
    #First, choose whether you'd like to view histograms or averages over time, then pick a semester and course to visualize.
    #''')
    page_select = st.segmented_control('Choose whether you\'d like to view histograms or averages over time', ['Histograms', 'Averages'], default='Histograms')
    if page_select == 'Histograms':
        st.write('#### Plot histograms of numerical responses.')
        
        #uncomment to make sure dataframe upload worked
        #st.dataframe(question_data)

    #############################
    ##### Begin making app ######
    #############################

        semester_list = [a[:2]+b[-2:] for a,b in unique_sorted(question_data['Semester'])] + ['All time']
        sem_select = st.selectbox("Please choose a semester.", semester_list,index=None)

        if sem_select is not None:

            sem_select_mapped = semester_mapper(sem_select)

            combine_toggle = st.toggle("Combine sections.",value=True)

            if sem_select != 'All time':
                course_list = unique_sorted(get_semester(question_data,sem_select_mapped)['Course Code'])
            else:
                course_list = unique_sorted(question_data['Course Code'])

            course_select = st.selectbox('Select a course.', course_list + ['All courses'], index=0)
            if course_select == 'All courses':
                course_select = None
            
            col1, col2, col3 = st.columns(3,vertical_alignment='bottom')
            cols = [col1, col2, col3]
            for i,col in enumerate(cols):

                with col:
                    #st.write(sem_select)
                    #question_select = st.selectbox("Please choose a question.", ('Question 1', 'Question 2', 'Question 3'),index=None)
                    

                    if (not combine_toggle):
                        
                        
                        fig_list = plot_hists_by_section(question_data, question_list[i], sem_select_mapped, course_select)
                        df_list = make_summary_df_section(question_data, question_list[i], sem_select_mapped, course_select)
                        st.write(f'**{question_list[i]}: {questions[question_list[i]]}**')
                        for fig,d in zip(fig_list,df_list):
                            st.write(fig)
                            st.dataframe(d,use_container_width=True)

                    elif (combine_toggle):
                        
                        fig_list = plot_hists_by_course(question_data, question_list[i], sem_select_mapped, course_select)
                        df_list = make_summary_df_course(question_data, question_list[i], sem_select_mapped, course_select)
                        st.write(f'**{question_list[i]}: {questions[question_list[i]]}**')
                        for fig,d in zip(fig_list,df_list):
                            st.write(fig)
                            st.dataframe(d,use_container_width=True)


    ## Figure out how or if to include the dataframe below:
    #    summary_df_list = []
    #    for sem in unique_sorted(question_data['Semester']):
    #        stylized_sem = ' '.join(sem)
    #        summary_df = get_semester(question_data,sem)[['Course Code','Question 1', 'Question 2', 'Question 3']].groupby('Course Code',sort=False).mean()
    #        summary_df.insert(0,'Semester',[stylized_sem]*len(summary_df))
    #        summary_df = summary_df.set_index(['Semester'],append=True).reorder_levels(['Semester','Course Code'])
    #
    #        summary_df_list = summary_df_list + [summary_df.round(2)]
    #
    #    summary = pd.concat(summary_df_list)
    #    st.dataframe(summary,width=100000)

########################### Averages over time plot ###############################
    elif page_select == 'Averages':

        st.write('#### Plot averages over time.')
        st.write('These are really intended to visualize student feedback for the same course over time. The plots are pretty boring for courses you\'ve only taught once.')

        course_select_avg = st.selectbox('Please choose a course.', 
                                         unique_sorted(question_data['Course Code'])+['Combine courses'], 
                                         placeholder='Please choose a course',
                                         index=None)
        plot_type_toggle = st.toggle('Toggle to plot as a bar graph.')
        plot_type = ['bar' if plot_type_toggle else 'line'][0]
        if course_select_avg is not None:
            #question_select_avg = st.selectbox("Please choose a question.", 
            #                                   question_list,
            #                                   index=0)

            col4, col5, col6 = st.columns(3,vertical_alignment='bottom')
            for i,col in enumerate([col4, col5, col6]):
                with col:
                    st.write(f'**{question_list[i]}: {questions[question_list[i]]}**')
                    if course_select_avg == 'Combine courses':
                        avg_fig = plot_averages(question_data, question_list[i], plot_type)
                    else:
                        avg_fig = plot_averages(question_data, question_list[i], plot_type, course_select_avg)
                    st.write(avg_fig)

                #plot the three question plots side by side