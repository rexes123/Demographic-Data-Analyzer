import pandas as pd

def calulate_demographic_data(print_data=True):

    # Read data from file
    df = pd.read_csv('adult.data.csv')
    totalData = len(df)
    
    # How many of each race are represented in this dataset? This should be a Pandas series with race names as the index labels.
    race_count = pd.Series(df['race'].value_counts())
    
    # What is the average age of men
    men_df =df[df['sex']== 'Male']
    average_age_men = round(men_df['age'].mean(), 10) 
    
    # What is the percentage of people who have a Bachelor's degree?
    percentage_bachelors = df['education']== "Bachelors"
    totalOfBachelors =  percentage_bachelors.sum()
    decimal = (totalOfBachelors/totalData)*100
    percentage = round(decimal, 10)
    
    # What percentage of people with advanced education (`Bachelors`, `Masters`, or `Doctorate`) make more than 50K?
    bachelors = df[(df['education'] == "Bachelors") & (df['salary'] == '>50K')]
    Masters = df[(df['education'] == "Masters") & (df['salary'] == '>50K')]
    Doctorate = df[(df['education'] == "Doctorate") & (df['salary'] == '>50K')]
    higher_education= len(bachelors) + len(Masters) + len(Doctorate)
    higher_education_rich=  higher_education/totalData 
    
    # What percentage of people without advanced education make more than 50K?
    lower_education =  df[(df['education'] != "Bachelors") &(df['education'] != "Masters") & (df['education'] != "Doctorate") & (df['salary'] == '>50K')]
    lower_education_rich = len(lower_education)/totalData
    
    # What is the minimum number of hours a person works per week (hours-per-week feature)?
    min_work_hours = df["hours-per-week"].min()
    
    # What percentage of the people who work the minimum number of hours per week have a salary of >50K?
    peo = df[(df['hours-per-week'].min()) & (df['salary'] == ">50K")]
    num_min_workers = len(peo)
    rich_percentage = round(((num_min_workers/len(df))*100), 10)
    
    # What country has the highest percentage of people that earn >50K?
    highEarners = df[df['salary'] == '>50K']
    totalCountry = df['native-country'].value_counts()
    highest_earning_country = (highEarners['native-country']).value_counts()
    percentageByCountry = (highest_earning_country / totalCountry)*100
    highest_earning_country = percentageByCountry.idxmax()
    highest_earning_country_percentage = round(percentageByCountry.max(), 10)
    
    # Identify the most popular occupation for those who earn >50K in India.
    highEarnersInIndia = df[(df['native-country']== 'India') & (df['salary']== '>50K')]
    top_IN_occupation = (highEarnersInIndia['occupation']).value_counts().idxmax()

    if print_data:
        print("Number of each race:\n", race_count) 
        print("Average age of men:", average_age_men)
        print(f"Percentage with Bachelors degrees: {percentage}%")
        print(f"Percentage with higher education that earn >50K: {higher_education_rich}%")
        print(f"Percentage without higher education that earn >50K: {lower_education_rich}%")
        print(f"Min work time: {min_work_hours} hours/week")
        print(f"Percentage of rich among those who work fewest hours: {rich_percentage}%")
        print("Country with highest percentage of rich:", highest_earning_country)
        print(f"Highest percentage of rich people in country: {highest_earning_country_percentage}%")
        print("Top occupations in India:", top_IN_occupation)




print(calulate_demographic_data(print_data=True))