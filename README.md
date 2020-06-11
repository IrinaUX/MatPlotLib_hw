### MatPlotLib_hw
Homework for MatPlotLib.

The top 4 most promising drug regimens from bar charts of Average Tumor Volume grouped by Drug Regimen are:
Capomulin, Ramicane, Infubinol and Ceftamin

There are 2 drug regimens with at least one outlier, among the top 4 most promising drugs identified earlier:
Capomulin and Ramicane

Moderate positive correlation is observed between Mouse Weight and the Average Tumor Volume:
0.53

#### Dependencies and Setup
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as sts
import numpy as np
import os

#### filter for warnings:
import warnings
warnings.filterwarnings('ignore')

#### File path for the saved plots
file_path = os.path.join("images","")

#### Study data files
mouse_metadata_path = "data/Mouse_metadata.csv"
study_results_path = "data/Study_results.csv"

#### Read the mouse data and the study results
mouse_metadata = pd.read_csv(mouse_metadata_path)
study_results = pd.read_csv(study_results_path)

#### Combine the data into a single dataset
merged_df = pd.merge(study_results, mouse_metadata, how='left', on='Mouse ID')
merged_df.head()

#### Checking the number of mice in the DataFrame.
len(merged_df['Mouse ID'].unique())

#### Getting the duplicate mice by ID number that shows up for Mouse ID and Timepoint. 
duplicated_mouse_df = merged_df.loc[merged_df.duplicated(subset=["Mouse ID","Timepoint"]), "Mouse ID"].unique()
duplicated_mouse_df

#### Optional: Get all the data for the duplicate mouse ID. 
duplicated_mouse = merged_df.loc[merged_df["Mouse ID"] == "g989"]
duplicated_mouse

#### Create a clean DataFrame by dropping the duplicate mouse by its ID.
clean_df = merged_df[merged_df["Mouse ID"].isin(duplicated_mouse_df) == False]
clean_df.head()

#### Checking the number of mice in the clean DataFrame.
len(clean_df["Mouse ID"].unique())

Summary Statistics
#### Generate a summary statistics table of mean, median, 
    ####variance, standard deviation, and SEM 
    ####of the tumor volume for each regimen
​
#### This method is the most straightforward, 
    ####creating multiple series and putting them all together at the end.
#### Create a summary table
means = clean_df.groupby("Drug Regimen").mean()["Tumor Volume (mm3)"]
medians = clean_df.groupby("Drug Regimen").median()["Tumor Volume (mm3)"]
variance = clean_df.groupby("Drug Regimen").var()["Tumor Volume (mm3)"]
sem = clean_df.groupby("Drug Regimen").sem()["Tumor Volume (mm3)"]
std = clean_df.groupby("Drug Regimen").std()["Tumor Volume (mm3)"]
means = clean_df.groupby("Drug Regimen").mean()["Tumor Volume (mm3)"]
summary_table1 = pd.DataFrame({"Average Tumor Volume (mm3)": means,
                               "Median Tumor Volume (mm3)": medians,
                               "Std. Dev. Tumor Volume (mm3)": std,
                               "Sem Tumor Volume (mm3)": sem,
                               "Variance Tumor Volume (mm3)": variance})
summary_table1

Average Tumor Volume (mm3)	Median Tumor Volume (mm3)	Std. Dev. Tumor Volume (mm3)	Sem Tumor Volume (mm3)	Variance Tumor Volume (mm3)
Drug Regimen					
Capomulin	40.675741	41.557809	4.994774	0.329346	24.947764
Ceftamin	52.591172	51.776157	6.268188	0.469821	39.290177
Infubinol	52.884795	51.820584	6.567243	0.492236	43.128684
Ketapril	55.235638	53.698743	8.279709	0.603860	68.553577
Naftisol	54.331565	52.509285	8.134708	0.596466	66.173479
Placebo	    54.033581	52.288934	7.821003	0.581331	61.168083
Propriva	52.320930	50.446266	6.622085	0.544332	43.852013
Ramicane	40.216745	40.673236	4.846308	0.320955	23.486704
Stelasyn	54.233149	52.431737	7.710419	0.573111	59.450562
Zoniferol	53.236507	51.818479	6.966589	0.516398	48.533355


Bar Plots
#### Generate a bar plot showing the number of mice per time point for each treatment throughout the course of the study using pandas.
#### Create a drug list to iterate over:
drug_list = summary_table1.index.tolist()
for x in range(len(drug_list)):
    drug_name = drug_list[x]
    drug_list[x] = clean_df.loc[clean_df["Drug Regimen"] == drug_list[x]]
    plot_title = f"{drug_name} - Mice Count vs. Timepoint"
    drug_list[x] = drug_list[x][["Mouse ID", "Timepoint"]]
    drug_list[x] = drug_list[x].groupby('Timepoint')
    mouse_id = drug_list[x]['Mouse ID'].count()
    timepoint_barchart = mouse_id.plot(kind='bar', title=plot_title, color='g')
    timepoint_barchart.set_xlabel("Timepoints")
    timepoint_barchart.set_ylabel("Mice Count")
    plt.show()

#### Same plots using plt.bar
x_axis = [value for value in range(0, 50, 5)]
#### Start with 'Capomulin'
drug_list_plt = summary_table1.index.tolist()
drug_name_plt = ""
for x in range(len(drug_list_plt)):
    drug_name_plt = drug_list_plt[x]
    print(drug_name_plt)
    
drug_list_plt = summary_table1.index.tolist()
drug_name_plt = ""
for x in range(len(drug_list)):
    drug_name_plt = drug_list_plt[x]
    ####print(drug_name_plt)
    plt.bar(x_axis, mouse_id, color='g', alpha=0.5, width=3, align="center")
    
    ####plt.xticks(value for value in range(0, 50, 5))
    plt.title(f"{drug_name_plt} - Mouse ID Count per Timepoint")
    plt.xlabel("Timepoints")
    plt.ylabel("Mouse ID Count")
    plt.savefig(f"{drug_name_plt}_MouseID_Count_per_Timepoint")
    plt.show()

#### Option 3 - aggregate the plots into one:
#### Create a grouped_df with two columns:
grouped_gf = clean_df.groupby(['Timepoint','Drug Regimen'])
mice_count = grouped_gf['Timepoint'].count()

#### Aggreate the data into one plot:
agg_df = clean_df.groupby(['Drug Regimen','Timepoint']).agg({'Mouse ID':'count'})

#### This dictionary is to color the bars in the custom plot based on specific drug.
values = ["royalblue","darkorange","green","red","lime","blueviolet","pink","cornflowerblue","peru","lightseagreen"]
keys = drug_list_plt
color_drug_dict = dict(zip(keys, values))
print(color_drug_dict)

#### Plot aggregated data
agg_df.unstack(0).plot(kind="bar",figsize=(12,3), label='_nolegend_', color=values, width=0.65) 
plt.title("Figure 1 - Aggregated results for Mouse ID Counts for each Drug Regimen at each Timepoint")
plt.legend(bbox_to_anchor=(1.05, 1))
plt.xlabel("Timepoints")
plt.ylabel("Mouse ID Count")
plt.tight_layout()
plt.show()
plt.savefig("agg_df_plot")

#### How to identify the 4 most promising drug regimens from the plot above.
#### Attempt 1 - Use sum() function on each of the drug regimen's timepoints
agg_tp_df = clean_df.groupby(['Drug Regimen', 'Timepoint']).agg({'Timepoint':'sum'})
print(agg_tp_df)

#### plot with unstack option:
agg_tp_df.unstack(0).plot(kind="bar",figsize=(12,3), label='_nolegend_', width=0.65) 
plt.legend(bbox_to_anchor=(1.05, 1))
plt.title("Figure 2 - Aggregated results for Sum of Timepoints for each Drug Regimen at each Timepoint")
plt.xlabel("Timepoints")
plt.ylabel("Timepoints Sum - Accumulative")
plt.tight_layout()
plt.show()
plt.savefig("agg_tp_df.png")
print(f"Based on the graph below, the top 4 treatments are: ")

#### Attempt 2 - Use max() function on each of the drug regimen's timepoints
agg_tp_max_df = clean_df.groupby(['Drug Regimen', 'Timepoint']).agg({'Timepoint':'max'})

#### plot with unstack option:
agg_tp_df.unstack(0).plot(kind="bar",figsize=(12,3), label='_nolegend_', width=0.65) 
plt.legend(bbox_to_anchor=(1.05, 1))
plt.title("Figure 3 - Aggregated results for Maximum Timepoints for each Drug Regimen at each Timepoint")
plt.xlabel("Timepoints")
plt.ylabel("Timepoints Max - Accumulative")
plt.tight_layout()
plt.show()
plt.savefig("agg_tp_df.png")
####print(f"Based on the graph below, the top 4 treatments are: ")

#### Try extracting top 4 Timepoints based on their maximum value:
mean_tumor_volume = clean_df.groupby(['Drug Regimen', 'Timepoint']).mean()
mean_tumor_volume.head()


Pie Plots

#### Generate a pie plot showing the distribution of female versus male mice using pandas
gender_df = clean_df.groupby("Sex")
gender_df.count()
gender_mice_filtered = gender_df["Sex"]
gender_mice_filtered.count()

#### Generate Pie Plot using plt.pie()
labels = ["Female Mice", "Male Mice"]
sizes = [gender_mice_filtered.count()]
colors = ['cyan', 'yellow']
plt.pie(sizes, labels=labels, autopct="%.2f%%", startangle=90, colors=colors)
plt.title("Mice Sex Distribution using plt.pie method")
plt.savefig("Female and Male Mice Distribution")
plt.show()

#### Sex distribution using plot.pie()
doc_df = pd.DataFrame({'Counts': [922, 958]},
                  index=['Female', 'Male'])
doc_df

plot_sex = doc_df.plot.pie(y='Counts', title="Mice Sex Distribution using plot.pie() method", labels=['Female','Male'], figsize=(4,4), startangle=90, autopct="%.2f%%", colors=['c','y'])


Quartiles, Outliers and Boxplots

#### Select data at last timepoint only. However some effective treatments might not have 45th day of treatment.
final_volume_df = clean_df.loc[clean_df['Timepoint'] == 45]

#### Top 4 most promising drugs could be obtained from the average tumor size:
    #### (wrong, see correct solution below) Find Top 4 most promising Drug Regimens for treating cancer in mice
grouped_final_volume_df = final_volume_df.groupby('Drug Regimen').mean()
grouped_final_volume_df.head(4)

#### Try again to find the top 4 by sorting the data by Tumor volume
sorted_grouped_final_vol_test7_df = final_volume_df.groupby(['Drug Regimen', 'Tumor Volume (mm3)'])
lambda_test7_df = sorted_grouped_final_vol_test7_df.apply(lambda x: x.sort_values('Tumor Volume (mm3)'))
lambda_test7_df.head(100)

#### FINAL - Find Top 4 most promising Drug Regimens for treating cancer in mice
sorted_grouped_final_vol_test7a_df = final_volume_df.groupby(['Drug Regimen'])

mean_sorted_grouped_final_vol_test7a_df = sorted_grouped_final_vol_test7a_df.mean()

#### FINAL - Find Top 4 most promising Drug Regimens for treating cancer in mice
summary_table2_Top4_MostPromisingDrugs = mean_sorted_grouped_final_vol_test7a_df.sort_values(by='Tumor Volume (mm3)', ascending=True, ignore_index=False)
summary_table2_Top4_MostPromisingDrugs.to_csv("Top4_Most_promising_drug_regimens_df.csv")
summary_table2_Top4_MostPromisingDrugs.head(4)

top4_treatment_regimens = summary_table2_Top4_MostPromisingDrugs.index[:4].to_list()
top4_treatment_regimens

#### README Instructions - Calculate the quartiles and IQR and 
    #### quantitatively determine if there are any potential outliers across all four treatment regimens.

#### Determine if there are any potential outliers in the drug regimen data
quartiles = clean_df['Tumor Volume (mm3)'].quantile([.25,.5,.75])
quartiles
lowerq = quartiles[0.25]
upperq = quartiles[0.75]
iqr = upperq-lowerq

print(f"The lower quartile of tumor volume is: {lowerq}")
print(f"The upper quartile of tumor volume is: {upperq}")
print(f"The interquartile range of tumor volume is: {iqr}")
print(f"The the median of tumor volume is: {quartiles[0.5]} ")

lower_bound = lowerq - (1.5*iqr)
upper_bound = upperq + (1.5*iqr)
print(f"Values below {lower_bound} could be outliers.")
print(f"Values above {upper_bound} could be outliers.")

outlier_tumor_volume = clean_df.loc[(clean_df['Tumor Volume (mm3)'] < lower_bound) \
                                        | (clean_df['Tumor Volume (mm3)'] > upper_bound)]
outlier_tumor_volume

grouped_outlier_tumor_volume = outlier_tumor_volume.groupby(['Drug Regimen', 'Mouse ID']).mean().reset_index()
grouped_outlier_tumor_volume.head(30)

top4_treatment_regimens = summary_table2_Top4_MostPromisingDrugs.index[:4].to_list()
top4_treatment_regimens

outlier_within_top4 = grouped_outlier_tumor_volume['Drug Regimen']
unique_outliers_arr = outlier_within_top4.unique()
unique_outliers_arr

outlier_within_top4 = []
for x in range(len(top4_treatment_regimens)):
    if top4_treatment_regimens[x] in unique_outliers_arr:
        print(x)
        outlier_within_top4.append(top4_treatment_regimens[x])
print(f"There are 2 drug regimens with at least one outlier, among the top 4 most promising drugs identified earlier:\
      {outlier_within_top4}")


#### README Instructions - Using Matplotlib, generate a box and whisker plot of the final tumor volume 
    #### for all four treatment regimens 
    #### and highlight any potential outliers in the plot by changing their color and style.

#### Prepare a new df to work on the statistics part
#### NOTE: for a single hard-coded drug at first:
tumor_volume_df = clean_df[['Mouse ID', 'Drug Regimen', 'Timepoint', 'Tumor Volume (mm3)']]
#### Grouped by the 'Drug Regimen' and "Timepoint"
grouped_tumor_volume_df = tumor_volume_df.groupby(['Drug Regimen', 'Timepoint'])
#### Reset index and sort values in descending order:
sorted_noindex_mean_grouped_tumor_volume_df = grouped_tumor_volume_df.mean().reset_index().sort_values(['Tumor Volume (mm3)'], ascending=False)
#### Try to use loc function to extract the data only for Capomulin drug regimen.
drug1_Capomulin = clean_df.loc[clean_df['Drug Regimen'] == 'Capomulin']
#### Select only Tumor Volume and Timpoint columns
drug1_Capomulin_filtered = drug1_Capomulin[['Timepoint', 'Tumor Volume (mm3)']]
#### Group by Timepoint
drug1_Capomulin_filtered_grouped = drug1_Capomulin_filtered.groupby('Timepoint')
#### Mean of group by object and rename the column of the filtered
drug1_Capomulin_filtered_grouped.mean()
drug1_Capomulin_filtered.rename(columns={'Tumor Volume (mm3)': 'Tumor Volume (mm3) Capomulin'})

#### From Instructions - README:
#### Calculate the final tumor volume of each mouse 
    #### across four of the most promising treatment regimens: 
        ####Capomulin, Ramicane, Infubinol, and Ceftamin. 
        
#### Calculate the quartiles and IQR and quantitatively determine 
    #### if there are any potential outliers across all four treatment regimens.

#### Create a single plot for Capomulin drug:    
tumor_volume = drug1_Capomulin_filtered['Tumor Volume (mm3)']
tumor_volume

fig1, ax1 = plt.subplots()
ax1.set_title('Tumor Volume (mm3) - Capomulin')
ax1.set_ylabel('Timepoints y axis')
ax1.boxplot(tumor_volume)
plt.show()

#### Create a for loop to iterate over drug list and generate box and whiskers plot for each of the drug regimens. Also .loc clean_df to only use data with Timepoint=45 (at the end of the treatment)
print(f"From the box and whiskers plots below, there are few outliers detected:")
print(f' - "Capomulin", "Propriva", "Ramicane", "Stelasyn"')
print('====================================================================================================')
print(f"Note: {outlier_within_top4} are listed in the top 4 most promising drugs, but both have outliers.")
print('====================================================================================================')

drug_list = summary_table1.index.tolist()
####print(drug_list)
for x in range(len(drug_list)):
    drug_name = drug_list[x]
    ####print(drug_name)
    drug_name_df = clean_df.loc[clean_df["Drug Regimen"] == drug_name]
    plot_title = f"{drug_name} - Mice Count vs. Timepoint"
    ####print(plot_title)
    filtered_drug_name_df = drug_name_df[['Timepoint', 'Tumor Volume (mm3)']]
####filetered_drug_name_df    
    grouped_filt_drug_name_df = filtered_drug_name_df.groupby('Timepoint')
####grouped_filt_drug_name_df
    mean_grouped_filt_drug_df = grouped_filt_drug_name_df.mean()
####mean_grouped_filt_drug_df    
    renamed_mean_grouped_df = mean_grouped_filt_drug_df.rename(columns={'Tumor Volume (mm3)': (f"Tumor Volume (mm3), {drug_name}")})
####renamed_mean_grouped_df
    #### option 2
    tumor_volume = drug_name_df['Tumor Volume (mm3)']
#### tumor_volume

    #### Create df based one the clean_df using .loc method to extract only the Timepoint == 45, as final tumor volume
    final_tumor_vol_df = drug_name_df.loc[clean_df['Timepoint'] == 45]
    grouped_final_tumor_vol_df = final_tumor_vol_df.groupby('Mouse ID')
####grouped_final_tumor_vol_df    
    mean_grouped_final_drug = grouped_final_tumor_vol_df.mean()
    renamed_mean_grouped_final_df = mean_grouped_final_drug.rename(columns={'Tumor Volume (mm3)': (f"Tumor Volume (mm3), {drug_name}"),
            'Timepoint': (f"Timepoint, {drug_name}"),
            'Metastatic Sites': (f"Metastatic Sites, {drug_name}"),
            'Age_months': (f"Age_months, {drug_name}"),
            'Weight (g)': (f"Weight (g), {drug_name}")})

####renamed_mean_grouped_final_df
    

    fig1, ax1 = plt.subplots()
    ax1.set_title(f'Tumor Volume (mm3) - {drug_name}')
    ax1.set_ylabel('Timepoints y axis')
    ax1.boxplot(tumor_volume)
    plt.savefig(f"Box_and_Whiskers_plot {drug_name}.png")
    plt.show()

#### README Instructions - All four box plots should be within the same figure. 
    #### Use this [Matplotlib documentation page]
        #### (https://matplotlib.org/gallery/pyplots/boxplot_demo_pyplot.html
        #### sphx-glr-gallery-pyplots-boxplot-demo-pyplot-py) 

#### test_box_plot for Ramicane ONLY:
print(top4_treatment_regimens)
box_plot_df = clean_df.loc[clean_df['Drug Regimen'] == outlier_within_top4[0]]
box_plot_df1 = box_plot_df['Tumor Volume (mm3)']
####box_plot_df = clean_df['Tumor Volume (mm3)']

drug_list = summary_table1.index.tolist()
####print(drug_list)
for x in range(len(top4_treatment_regimens)):
    drug_name = top4_treatment_regimens[x]
    ####print(drug_name)
    drug_name_df = clean_df.loc[clean_df["Drug Regimen"] == drug_name]
    plot_title = f"{drug_name} - Mice Count vs. Timepoint"
    ####print(plot_title)
    filtered_drug_name_df = drug_name_df[['Timepoint', 'Tumor Volume (mm3)']]
####filetered_drug_name_df    
    grouped_filt_drug_name_df = filtered_drug_name_df.groupby('Timepoint')
####grouped_filt_drug_name_df
    mean_grouped_filt_drug_df = grouped_filt_drug_name_df.mean()
####mean_grouped_filt_drug_df    
    renamed_mean_grouped_df = mean_grouped_filt_drug_df.rename(columns={'Tumor Volume (mm3)': (f"Tumor Volume (mm3), {drug_name}")})
####renamed_mean_grouped_df
    #### option 2
    tumor_volume = drug_name_df['Tumor Volume (mm3)']
#### tumor_volume

    #### Create df based one the clean_df using .loc method to extract only the Timepoint == 45, as final tumor volume
    final_tumor_vol_df = drug_name_df.loc[clean_df['Timepoint'] == 45]
    grouped_final_tumor_vol_df = final_tumor_vol_df.groupby('Mouse ID')
####grouped_final_tumor_vol_df    
    mean_grouped_final_drug = grouped_final_tumor_vol_df.mean()
    renamed_mean_grouped_final_df = mean_grouped_final_drug.rename(columns={'Tumor Volume (mm3)': (f"Tumor Volume (mm3), {drug_name}"),
                'Timepoint': (f"Timepoint, {drug_name}"),
                'Metastatic Sites': (f"Metastatic Sites, {drug_name}"),
                'Age_months': (f"Age_months, {drug_name}"),
                'Weight (g)': (f"Weight (g), {drug_name}")})

####renamed_mean_grouped_final_df 

    fig1, ax1 = plt.subplots()
    ax1.set_title(f'Tumor Volume (mm3) - {drug_name}')
    ax1.set_ylabel('Timepoints y axis')
    ax1.boxplot(tumor_volume)
    plt.savefig(f"Box_and_Whiskers_plot {drug_name}.png")
    plt.show()

tumor_vol_ramicane = clean_df.loc[clean_df['Drug Regimen'] == 'Ramicane']
tumor_vol_capomulin = clean_df.loc[clean_df['Drug Regimen'] == 'Capomulin']
tumor_vol_ceftamin = clean_df.loc[clean_df['Drug Regimen'] == 'Ceftamin']
tumor_vol_infubinol = clean_df.loc[clean_df['Drug Regimen'] == 'Infubinol']
#### tumor_vol_ramicane_arr_of_lists = []
#### for x in range(len(top4_treatment_regimens)):
tumor_vol_ramicane = tumor_vol_ramicane[['Tumor Volume (mm3)']]
tumor_vol_capomulin = tumor_vol_capomulin[['Tumor Volume (mm3)']]
tumor_vol_ceftamin = tumor_vol_ceftamin[['Tumor Volume (mm3)']]
tumor_vol_infubinol = tumor_vol_infubinol[['Tumor Volume (mm3)']]

tumor_vol_ramicane_list = tumor_vol_ramicane['Tumor Volume (mm3)'].tolist()
tumor_vol_capomulin_list = tumor_vol_capomulin['Tumor Volume (mm3)'].tolist()
tumor_vol_ceftamin_list = tumor_vol_ceftamin['Tumor Volume (mm3)'].tolist()
tumor_vol_infubinol_list = tumor_vol_infubinol['Tumor Volume (mm3)'].tolist()

data = [tumor_vol_ramicane_list, tumor_vol_capomulin_list, tumor_vol_ceftamin_list, tumor_vol_infubinol_list]
fig5, ax5 = plt.subplots()
ax5.set_title(f'Top 4 most promising treatments, their comparable effectiveness and existing outliers')
####ax5.set_xlabel("Top 4 most promising treatments")
ax5.set_ylabel('Tumor Volume (mm3)')
ax5.boxplot(data)
labels = ('Ramicane', 'Capomulin', 'Ceftamin', 'Infubinol')
names=("Ramicane","Capomulin","Ceftamin", "Infubinol")
plt.xticks(np.arange(len(labels))+1, labels)
####plt.savefig(f"Box_and_Whiskers_plot Ramicane.png")
plt.show()

#### Generate a line plot of time point versus tumor volume 
    #### for a mouse treated with Capomulin
mouse_treated_with_capomulin = clean_df.loc[clean_df['Drug Regimen'] == 'Capomulin']
mice_capomulin = mouse_treated_with_capomulin[['Mouse ID', 'Drug Regimen', 'Tumor Volume (mm3)']]
unique_mouse = mice_capomulin['Mouse ID'].unique()
####😍👌☺️😊
unique_mouse
#### Chose mouse ID b128:
mouse_b128 = mouse_treated_with_capomulin.loc[mouse_treated_with_capomulin['Mouse ID'] == 'b128']
mouse_b128
mouse_b128_filtered = mouse_b128[['Mouse ID', 'Timepoint', 'Tumor Volume (mm3)']]
mouse_b128_filtered
mouse_b128_timepoint_list = mouse_b128_filtered['Timepoint'].tolist()
mouse_b128_timepoint_list
mouse_b128_tumvol_list = mouse_b128_filtered['Tumor Volume (mm3)'].tolist()
mouse_b128_tumvol_list
plt.plot(mouse_b128_timepoint_list, mouse_b128_tumvol_list)
plt.xlabel

#### plt.legend(bbox_to_anchor=(1.05, 1))
plt.title("Figure 4 - Mouse ID b128 - Timepoint versus Tumor Volume (mm3)")
plt.xlabel("Timepoints (Days)")
plt.ylabel("Tumor Volume (mm3)")
####plt.tight_layout()
plt.show()
plt.savefig("Mouse_id_b128_Timepoint_vs_Tumor_volume.png")

#### Generate a scatter plot of mouse weight versus average tumor volume for the Capomulin regimen
#### README:
#### Generate a scatter plot of mouse weight versus average tumor volume for the Capomulin treatment regimen.
capomulin_df = clean_df.loc[clean_df['Drug Regimen'] == 'Capomulin']
mouse_b128_cap_df = capomulin_df.loc[capomulin_df['Mouse ID'] == 'b128']
mouse_b128_cap_df

mouse_b128_cap_df2 = mouse_b128_cap_df[['Weight (g)', 'Tumor Volume (mm3)']]
mouse_b128_cap_df2 = mouse_b128_cap_df.groupby('Tumor Volume (mm3)').mean()
mouse_b128_cap_df

weight_b128_cap = mouse_b128_cap_df2['Weight (g)'].tolist()
print(weight_b128_cap)

#### weight_list_capomulin
average_b128_cap_vol = mouse_b128_cap_df['Tumor Volume (mm3)'].tolist()
average_b128_cap_vol


Correlation and Regression

#### Calculate the correlation coefficient and linear regression model 
#### for mouse weight and average tumor volume for the Capomulin regimen
capomulin_df3 = clean_df.loc[clean_df['Drug Regimen'] == 'Capomulin']
mouse_b128_cap_df3 = capomulin_df.loc[capomulin_df['Mouse ID'] == 'b128']
mouse_b128_cap_df3.head()

correlation = sts.pearsonr(average_b128_cap_vol, weight_b128_cap)
print(f"Correlation between both factors is {correlation}")

plt.scatter(average_b128_cap_vol, weight_b128_cap)
plt.title("Mouse ID b128 Weight vs. Average Tumor Volume (mm3)")
plt.xlabel("Tumor Volume (mm3)")
plt.ylabel("Mouse ID b128 Weight (g)")
plt.show()

capomulin_df4 = clean_df[clean_df['Drug Regimen'] == 'Capomulin']
capomulin_df4

mouse_weight_capomulin_df4 = capomulin_df4[['Weight (g)', 'Tumor Volume (mm3)']]
mouse_weight_capomulin_df4

#### View in scatter plot before calculating the correlation
plt.scatter(mouse_weight_capomulin_df4.iloc[:,0], mouse_weight_capomulin_df4.iloc[:,1])
plt.title("Capomulin - Mouse Weight vs. Tumor Volume")
plt.xlabel('Mouse Weight (g)')
plt.ylabel('Tumor Volume (mm3)')
plt.xlim(15, 25)
plt.ylim(36, 46)
plt.show()

#### Calculate the correlation coefficient - Mouse weight versus Tumor Vol
mouse_weight_capomulin = mouse_weight_capomulin_df4.iloc[:,0]
tumor_vol = mouse_weight_capomulin_df4.iloc[:,1]
correlation4 = sts.pearsonr(mouse_weight_capomulin, tumor_vol)
print(correlation4)

#### Calculate the correlation coefficient - Mouse weight versus 
    #### AVERAGE Tumor Vol. To get average(mean()) need to create groupby 
grouped_df5 = mouse_weight_capomulin_df4.groupby('Weight (g)')
mean_grouped_df5 = grouped_df5.mean().reset_index()
mean_grouped_df5

#### Plot Mouse Weight versus AVERAGE Tumor Volume
mouse_weight_capomulin5 = mean_grouped_df5.iloc[:,0]
tumor_vol_average5 = mean_grouped_df5.iloc[:,1]
correlation5 = sts.pearsonr(mouse_weight_capomulin5, tumor_vol_average5)
print(correlation4)

#### View in scatter plot before calculating the correlation
plt.scatter(mean_grouped_df5.iloc[:,0], mean_grouped_df5.iloc[:,1])
plt.title("Capomulin - Mouse Weight vs. Average Tumor Volume")
plt.xlabel('Mouse Weight (g)')
plt.ylabel('Average Tumor Volume (mm3)')
plt.xlim(15, 25)
plt.ylim(36, 46)
plt.show()

#### Try to use unstack method to get the correlation for mouse_weight_capomulin_df4
####mouse_weight_capomulin_df4
mouse_weight_capomulin_df4 = mouse_weight_capomulin_df4.rename(columns={
            'Weight (g)': 'Weight',
            'Tumor Volume (mm3)': 'TumorVolume'})
mouse_weight_capomulin_df4

weight6 = mouse_weight_capomulin_df4.Weight
tumorvol = mouse_weight_capomulin_df4.TumorVolume
plt.scatter(weight6, tumorvol)
plt.title("Test 6 - Mouse Weight vs. Tumor Volume")
plt.xlabel("Weight (g)")
plt.ylabel("Tumor Volume (mm3)")
plt.xlim(15, 25)
plt.ylim(36, 46)
plt.show()

round(sts.pearsonr(weight6, tumorvol)[0],2)

mouse_corr = mouse_weight_capomulin_df4.corr()
mouse_corr.unstack().sort_values()

#### README - lot the linear regression model 
    #### on top of the previous scatter plot.
#### create numpy array from df
weight7 = mean_grouped_df5['Weight (g)'].to_numpy()
ave_tum_vol = mean_grouped_df5['Tumor Volume (mm3)'].to_numpy()
print(weight7)
print(ave_tum_vol)

#### Now plot
####average_tumorvol = mean_grouped_df5 #### .iloc[:,1]
plt.plot(weight7, ave_tum_vol, 'o')

m, b = np.polyfit(weight7, ave_tum_vol, 1)
plt.plot(weight7, m*weight7 + b)
plt.grid()
plt.xlim(15, 25)
plt.ylim(36, 46)
plt.title("Test 7 FINAL - Mouse Weight vs. Average Tumor Volume")
plt.xlabel("Weight (g)")
plt.ylabel("Average Tumor Volume (mm3)")
plt.show()

