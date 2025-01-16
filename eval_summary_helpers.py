import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import streamlit as st
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from matplotlib.colors import ListedColormap
import warnings
warnings.filterwarnings('ignore')

def sort_survey_data(df):

    '''
    Sorts the survey data by (1) semester, (2) course code, and (3) section

    Arguments:
        df - DataFrame containing survey response data
    Returns:
        sorted_df - DataFrame sorted as stated above
    '''
    sems = ['Fall', 'Summer', 'Spring']
#    sorted_df = df.sort_values(by='Semester', kind='stable', ascending=False, key=lambda sem: sem.apply(lambda x: x[1]))
#    sorted_df = sorted_df.sort_values(by='Semester', kind='stable', ascending=True, key=lambda sem: sem.apply(lambda x: sems.index(x[0])))
    
    sorted_df = df.sort_values(by='Section', kind='stable', ascending=True)
    sorted_df = sorted_df.sort_values(by='Course Code', kind='stable', ascending=True)
    sorted_df = sorted_df.sort_values(by='Semester', kind='stable', ascending=True, key=lambda sem: sem.apply(lambda x: sems.index(x[0])))
    sorted_df = sorted_df.sort_values(by='Semester', kind='stable', ascending=False, key=lambda sem: sem.apply(lambda x: x[1]))

    return sorted_df


def make_hist_plots(df, column):

    '''
    Makes a histrogram from numerical survey data (Questions 1, 2, or 3)
    Arguments:
        df - DataFrame containing survey response data
        column - the column name of the df to be plotted. Must be 'Question 1', 'Question 2', or 'Question 3'
    Returns:
        fig, ax - Figure and Axis objects containing the plotted histogram
        column - same as the input, for convenience
    '''

    assert column in ['Question 1', 'Question 2', 'Question 3'], 'column must be \'Question 1\', \'Question 2\', or \'Question 3\''
    
    # Define the bins
    bins = np.arange(1, 6)  # Bins from 1 to 5
    hist, bin_edges = np.histogram(df[column], bins=bins)

    #get max y tick mark
    max_list = []
    for q in ['Question 1', 'Question 2', 'Question 3']:
        hist_temp, _ = np.histogram(df[q], bins=bins)
        max_list = max_list + [hist_temp.max()]
    
    fig, ax = plt.subplots()
    plt.bar(bin_edges[:-1], hist, width=0.8, align='center')
    plt.xticks(bin_edges[:-1])  # Set x-axis ticks to match the bins
    y_max = max(max_list)
    if y_max <= 6:
        plt.yticks(np.linspace(0,y_max,y_max+1).round())
        plt.ylim((0,y_max+0.5))
    else:
        plt.yticks(np.linspace(0,y_max,6).round())
        plt.ylim((0,y_max+1))

    return fig, ax, column


def get_semester(df, semester):
    '''
    Get all survey response data for a particular semester
    Arguments:
        df - dataframe containing evaluation data to be visualized
        semester - the semester to be visualized, input as ('Season', 'year'); e.g., ('Fall', '2024'). If None, plot over all semesters.
    '''
    
    return df[df['Semester']==semester]


def get_course(df, course):
    '''
    Get all survey response data for a particular course
    Arguments:
        df - dataframe containing evaluation data to be visualized
        course - the course to be visualized, input as MATHXXXX; e.g., MATH4381
    '''

    return df[df['Course Code']==course]


def unique_sorted(seq):
    '''
    Return all unique elements from an iterable while maintaining the order those elements appear in the iterable

    Arguments:
        seq - iterable from which unique elements are to be extracted
    Returns:
        list of unique elements in the order they appear for the first time in the original iterable
    '''
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def plot_hists_by_section(df, column, semester=None, course=None):
    '''
    Plots histograms of all sections taught in a semester or all time

    Arguments:
        df - dataframe containing evaluation data to be visualized
        column - column of the dataframe to be plotted as a histogram
        semester - the semester to be visualized, input as ('Season', 'year'); e.g., ('Fall', '2024'). If None, plot over all semesters.
        course - plot all sections of the given course, input as MATHXXXX; e.g., MATH4381. If None, plot over all courses in the given semester.
    '''
    fig_list = []
    if semester:
        
        semester_df = get_semester(df,semester)
        if course:

            course_semester_df = get_course(semester_df, course)
            for sec in unique_sorted(course_semester_df['Section']):
                sec_df = course_semester_df[(course_semester_df['Section']==sec) & (course_semester_df['Course Code']==course)]
                fig, ax, _ = make_hist_plots(sec_df, column)
                ax.set_title(column + f' (n = {str(len(sec_df))})', fontsize=14)
                fig.suptitle(' '.join(semester) + ', ' + course + ' Section ' + sec, fontsize=14)
                ax.set_xlabel('Rating', fontsize=14)
                ax.set_ylabel(f'Count', fontsize=14)

                fig_list = fig_list + [fig]

        else:

            for course in unique_sorted(semester_df['Course Code']):
                course_semester_df = get_course(semester_df, course)
                for sec in unique_sorted(course_semester_df['Section']):
                    sec_df = course_semester_df[(course_semester_df['Section']==sec) & (course_semester_df['Course Code']==course)]
                    fig, ax, _ = make_hist_plots(sec_df, column)
                    ax.set_title(column + f' (n = {str(len(sec_df))})', fontsize=14)
                    fig.suptitle(' '.join(semester) + ', ' + course + ' Section ' + sec, fontsize=14)
                    ax.set_xlabel('Rating', fontsize=14)
                    ax.set_ylabel(f'Count', fontsize=14)

                    fig_list = fig_list + [fig]
                
    else:

        if course:
            for sem in unique_sorted(df['Semester']):
                semester_df = get_semester(df,sem)
                course_semester_df = get_course(semester_df, course)
                for sec in unique_sorted(course_semester_df['Section']):
                    sec_df = course_semester_df[(course_semester_df['Section']==sec) & (course_semester_df['Course Code']==course)]
                    fig, ax, _ = make_hist_plots(sec_df, column)
                    ax.set_title(column + f' (n = {str(len(sec_df))})', fontsize=14)
                    fig.suptitle(' '.join(sem) + ', ' + course + ' Section ' + sec, fontsize=14)
                    ax.set_xlabel('Rating', fontsize=14)
                    ax.set_ylabel(f'Count', fontsize=14)

                    fig_list = fig_list + [fig]
        else:

            for sem in unique_sorted(df['Semester']):
                semester_df = get_semester(df,sem)
                for course in unique_sorted(df['Course Code']):
                    course_semester_df = get_course(semester_df, course)
                    for sec in unique_sorted(course_semester_df['Section']):
                        sec_df = course_semester_df[(course_semester_df['Section']==sec) & (course_semester_df['Course Code']==course)]
                        fig, ax, _ = make_hist_plots(sec_df, column)
                        ax.set_title(column + f' (n = {str(len(sec_df))})', fontsize=14)
                        fig.suptitle(' '.join(sem) + ', ' + course + ' Section ' + sec, fontsize=14)
                        ax.set_xlabel('Rating', fontsize=14)
                        ax.set_ylabel(f'Count', fontsize=14)

                        fig_list = fig_list + [fig]
                    
    return fig_list

def plot_hists_by_course(df, column, semester=None, course=None):
    '''
    Plot histograms of all courses taught in a semester or all time (that is, combine all sections of the same course into a single histogram)
    Arguments:
        df - dataframe containing evaluation data to be visualized
        column - column of the dataframe to be plotted as a histogram
        semester - the semester to be visualized, input as ('Season', 'year'); e.g., ('Fall', '2024'). If None, plot over all semesters.
        course - the course to be visualized, input as MATHXXXX; e.g., MATH4381. If None, plot over all courses in the given semester.
    '''

    fig_list = []
    if semester:

        semester_df = get_semester(df,semester)

        if course:

            course_semester_df = get_course(semester_df, course)
            fig, ax, _ = make_hist_plots(course_semester_df, column)
            ax.set_title(column + f' (n = {str(len(course_semester_df))})', fontsize=14)
            fig.suptitle(' '.join(semester) + ', ' + course, fontsize=14)
            ax.set_xlabel('Rating', fontsize=14)
            ax.set_ylabel(f'Count', fontsize=14)

            fig_list = fig_list + [fig]
        else:

            #for c in unique_sorted(semester_df['Course Code']):
            #    
            #    course_semester_df = get_course(semester_df, c)
            #    fig, ax, _ = make_hist_plots(course_semester_df, column)
            #    ax.set_title(column + f' (n = {str(len(course_semester_df))})', fontsize=14)
            #    fig.suptitle(' '.join(semester) + ', ' + c, fontsize=14)
            #    ax.set_xlabel('Rating', fontsize=14)
            #    ax.set_ylabel(f'Count', fontsize=14) 
            #
            #    fig_list = fig_list + [fig]
            fig, ax, _ = make_hist_plots(semester_df, column)
            ax.set_title(column + f' (n = {str(len(semester_df))})', fontsize=14)
            fig.suptitle(' '.join(semester) + ', all courses', fontsize=14)
            ax.set_xlabel('Rating', fontsize=14)
            ax.set_ylabel(f'Count', fontsize=14)

            fig_list = fig_list + [fig]
    else:

        if course:

            course_df = get_course(df, course)
            fig, ax, _ = make_hist_plots(course_df, column)
            ax.set_title(column + f' (n = {str(len(course_df))})', fontsize=14)
            fig.suptitle('All time, ' + course, fontsize=14)
            ax.set_xlabel('Rating', fontsize=14)
            ax.set_ylabel(f'Count', fontsize=14) 

            fig_list = fig_list + [fig]

        else:
            
            #for c in unique_sorted(df['Course Code']):
            #    
            #    course_df = get_course(df, c)
            #    fig, ax, _ = make_hist_plots(course_df, column)
            #    ax.set_title(column + f' (n = {str(len(course_df))})', fontsize=14)
            #    fig.suptitle('All time, ' + c, fontsize=14)
            #    ax.set_xlabel('Rating', fontsize=14)
            #    ax.set_ylabel(f'Count', fontsize=14) 
            #
            #    fig_list = fig_list + [fig]
            fig, ax, _ = make_hist_plots(df, column)
            ax.set_title(column + f' (n = {str(len(df))})', fontsize=14)
            fig.suptitle('All courses, all time', fontsize=14)
            ax.set_xlabel('Rating', fontsize=14)
            ax.set_ylabel(f'Count', fontsize=14)

            fig_list = fig_list + [fig]

    return fig_list


def plot_averages(df, column, plot_type='bar',course=None):

    '''
    Plot bar charts of the average numerical score for any survey question over time
    Arguments:
        df - dataframe containing evaluation data to be visualized
        column - column of the dataframe to be plotted. Must be 'Question 1', 'Question 2', or 'Question 3'
        course - the course to be visualized, input as MATHXXXX; e.g., MATH4381. If None, plot over all courses.
    '''
        
    if course:
        gb_mean = get_course(df,course)\
                [['Semester','Question 1', 'Question 2', 'Question 3']].groupby('Semester',sort=False).mean()
        gb_count = get_course(df,course)\
                [['Semester','Question 1', 'Question 2', 'Question 3']].groupby('Semester',sort=False).count()

        xticks = np.arange(len(gb_mean))
        xtick_labels = [a[:2]+b[-2:]+f'\n (n={gb_count[column].loc[[(a,b)]][0]})' for a,b in gb_mean.index]
        xtick_labels.reverse()
        yticks = np.arange(5)
        fig, ax = plt.subplots()
        if plot_type=='bar':
            plt.bar(xticks, gb_mean[column].values, width=0.4)#, '-o', linewidth=2, markersize=10)
            for x,y in enumerate(gb_mean[column].values):
                plt.text(x,y+.3,f'{y:.3}',fontsize=14,ha='center',va='top')
                #plt.text(x,y-.25,f'{y:.3}\n'+'n='+str(gb_count[column].iloc[x]),fontsize=14,c='w',ha='center',va='top',weight='bold')
                #plt.text(x+.025,y,'n='+str(gb_count['Question 1'].iloc[x]),fontsize=12)
        elif plot_type=='line':
            plt.plot(xticks, gb_mean[column].values, '-o', linewidth=3, markersize=14)
            for x,y in enumerate(gb_mean[column].values):
                plt.text(x,y+.5,f'{y:.3}',fontsize=14,ha='center',va='top')
                #plt.text(x,y-.25,f'{y:.3}\n'+'n='+str(gb_count[column].iloc[x]),fontsize=14,c='w',ha='center',va='top',weight='bold')
                #plt.text(x+.025,y,'n='+str(gb_count['Question 1'].iloc[x]),fontsize=12)
        plt.ylim(0, 4)
        plt.xticks(xticks, xtick_labels,fontsize=14)
        plt.yticks(yticks, fontsize=14)
        ax.set_title(column,fontsize=14)
        fig.suptitle(course,fontsize=14)
        ax.set_ylim(0,5)
        if len(gb_mean)==1:
            ax.set_xlim(-.6, .6)

    
    else:
        gb_mean = df[['Semester','Question 1', 'Question 2', 'Question 3']].groupby('Semester',sort=False).mean()
        gb_count = df[['Semester','Question 1', 'Question 2', 'Question 3']].groupby('Semester',sort=False).count()

        xticks = np.arange(len(gb_mean))
        xtick_labels = [a[:2]+b[-2:]+f'\n (n={gb_count[column].loc[[(a,b)]][0]})' for a,b in gb_mean.index]
        xtick_labels.reverse()
        yticks = np.arange(5)
        fig, ax = plt.subplots()
        if plot_type=='bar':
            plt.bar(xticks, gb_mean[column].values, width=0.4)#, '-o', linewidth=2, markersize=10)
            for x,y in enumerate(gb_mean[column].values):
                plt.text(x,y+.3,f'{y:.3}',fontsize=14,ha='center',va='top')
                #plt.text(x,y+.6,f'{y:.3}\n'+'n='+str(gb_count[column].iloc[x]),fontsize=14,ha='center',va='top')
                #plt.text(x+.025,y,'n='+str(gb_count['Question 1'].iloc[x]),fontsize=12)
        elif plot_type=='line':
            plt.plot(xticks, gb_mean[column].values, '-o', linewidth=3, markersize=14)
            for x,y in enumerate(gb_mean[column].values):
                plt.text(x,y+.5,f'{y:.3}',fontsize=14,ha='center',va='top')
                #plt.text(x,y-.25,f'{y:.3}\n'+'n='+str(gb_count[column].iloc[x]),fontsize=14,c='w',ha='center',va='top',weight='bold')
                #plt.text(x+.025,y,'n='+str(gb_count['Question 1'].iloc[x]),fontsize=12)
        plt.ylim(0, 4)
        #plt.xticks(xticks, [' '.join(sem) for sem in gb_mean.index],fontsize=14)
        plt.xticks(xticks, xtick_labels,fontsize=14)
        plt.yticks(yticks, fontsize=14)
        ax.set_title(column,fontsize=14)
        fig.suptitle('All courses',fontsize=14)
        ax.set_ylim(0,5)
        if len(gb_mean)==1:
            ax.set_xlim(-.6, .6)
    return fig

def semester_mapper(sem):
    '''
    Takes in a string of the form YYXX, where YY is the season abbreviation and XX is the year of a semester; e.g., Fa24.
    Arguments:
        sem - string of the form defined above
    Returns:
        tuple of the form (Season,Year); e.g., ('Fall','2024')
    '''

    sem_list = ['Fa', 'Su', 'Sp']
    #assert sem[:2] in sem_list, 'Semester must be Fa, Su, or Sp'
    #assert len(sem)==4, 'Semeset must be of the form YYXX; e.g., Fa24'

    year = '20'+sem[-2:]

    if sem[:2]=='Fa':
        semester_tuple = ('Fall', year)

    if sem[:2]=='Su':
        semester_tuple = ('Summer', year)

    if sem[:2]=='Sp':
        semester_tuple = ('Spring', year)

    if sem == 'All time':
        semester_tuple = None
    
    return semester_tuple

def concat_dfs(df_list):
   
    question_data_list = []
    for f in df_list:
        
        # read the Excel file 
        eval_data = pd.read_excel(f) 
        
        instructor_name = eval_data['InstructorName']
        course_code = eval_data['CourseCode'][0].split('.')[2]
        course_title_list = eval_data['CourseTitle'][0].split()
        course_name = ' '.join(course_title_list[:course_title_list.index('Section')])
        semester = tuple(course_title_list[course_title_list.index('Section')+2:course_title_list.index('Section')+5:2])
        section = course_title_list[course_title_list.index('Section')+1]

        question_data_temp = eval_data.iloc[:,-7:]

        length = len(question_data_temp)

        question_data_temp.insert(0, 'Section', [section]*length)
        question_data_temp.insert(0, 'Semester', [semester]*length)
        question_data_temp.insert(0, 'Course Name', [course_name]*length)
        question_data_temp.insert(0, 'Course Code', [course_code]*length)
        question_data_temp.insert(0, 'Instructor Name', instructor_name)

        is_same = 0
        for d in question_data_list:
            is_same += question_data_temp.equals(d)
        
        if is_same == 0:
            question_data_list = question_data_list + [question_data_temp]

    return pd.concat(question_data_list)

def make_summary_df(df,column):
    
    #qs = ['Question 1', 'Question 2', 'Question 3']
    scores = [1,2,3,4]

    df_counts = pd.DataFrame(columns=[column], index=scores)
    df_counts[column] = df[column].value_counts()
    #for q in qs:
    #    #print(df[q].value_counts())
    #    df_counts[q] = df[q].value_counts()

    df_counts = df_counts.fillna(0).astype(int).transpose()
    avgs = np.dot(df_counts,[1,2,3,4])/df_counts.sum(1)
    
    df_counts['n'] = df_counts[[1,2,3,4]].sum(1)
    df_counts['Avg'] = avgs
    return df_counts.round(2)

def make_summary_df_course(df, column, semester=None, course=None):


    df_list = []
    if semester:

        semester_df = get_semester(df,semester)

        if course:

            course_semester_df = get_course(semester_df, course)
            df_list = df_list + [make_summary_df(course_semester_df, column)]
        else:

            df_list = df_list + [make_summary_df(semester_df, column)]
    else:

        if course:

            course_df = get_course(df, course)
            df_list = df_list + [make_summary_df(course_df, column)]

        else:
            
            df_list = df_list + [make_summary_df(df, column)]

    return df_list

def make_summary_df_section(df, column, semester=None, course=None):

    df_list = []
    if semester:
        
        semester_df = get_semester(df,semester)
        if course:

            course_semester_df = get_course(semester_df, course)
            for sec in unique_sorted(course_semester_df['Section']):
                sec_df = course_semester_df[(course_semester_df['Section']==sec) & (course_semester_df['Course Code']==course)]
                df_list = df_list + [make_summary_df(sec_df, column)]

        else:

            for course in unique_sorted(semester_df['Course Code']):
                course_semester_df = get_course(semester_df, course)
                for sec in unique_sorted(course_semester_df['Section']):
                    sec_df = course_semester_df[(course_semester_df['Section']==sec) & (course_semester_df['Course Code']==course)]

                    df_list = df_list + [make_summary_df(sec_df, column)]
                
    else:

        if course:
            for sem in unique_sorted(df['Semester']):
                semester_df = get_semester(df,sem)
                course_semester_df = get_course(semester_df, course)
                for sec in unique_sorted(course_semester_df['Section']):
                    sec_df = course_semester_df[(course_semester_df['Section']==sec) & (course_semester_df['Course Code']==course)]

                    df_list = df_list + [make_summary_df(sec_df, column)]
        else:

            for sem in unique_sorted(df['Semester']):
                semester_df = get_semester(df,sem)
                for course in unique_sorted(df['Course Code']):
                    course_semester_df = get_course(semester_df, course)
                    for sec in unique_sorted(course_semester_df['Section']):
                        sec_df = course_semester_df[(course_semester_df['Section']==sec) & (course_semester_df['Course Code']==course)]

                        df_list = df_list + [make_summary_df(df, column)]
                    
    return df_list