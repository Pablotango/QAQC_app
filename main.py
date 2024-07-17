import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import base64  # Import the base64 module
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go



# Function to load CSV file and return a DataFrame

def load_data(file):
    df = pd.read_csv(file)
    return df


def intertek_clean(df):
    '''This function cleans an Intertek csv by:
        changing column names,
        adds a column with Batch No,
        clean negatives'''
    # make a copy of the original df to avoid changing df0#
    df = df.copy()
    
    ##----##
    # Change the name of the columns so Ag becomes Ag_ppm....
    elem_list = list(df.columns)
    elem_list = [item + "_" for item in elem_list ] #Adds _ to every element

    unit_list = df.iloc[0].tolist()
    unit_list = [str(i) for i in unit_list] # convert all elements to string
    unit_list = [word.replace('%' , 'pct') for word in unit_list]

    header_list = [f'{x}{y}' for x, y in zip(elem_list, unit_list)]
    df.columns = header_list
    df.rename(columns = {'ELEMENTS_UNITS' : 'SampleID'}, inplace = True)
    
    # Add a column called 'Batch' that will contain the file name source
    # new_batch = ''
    # df.insert(0,'Batch',new_batch)
    
    #Delete first rows (metadata)
    
    df = df.iloc[6:,:]
    
    # Convert all to numeric, treat negatives and replace '<'
    
    for i in df.columns[1:]: # Start from the second column
    # Convert the column to numeric
           df[i] = pd.to_numeric(df[i], errors='coerce')
                    
           # Select the negative values and apply the transformation
           df.loc[df[i] < 0, i] = abs(df[i]) / 2
           
           # Replace non-digit characters by ''
           df.replace(r'\D', '', regex=True)
    
    # Restart index
    df.reset_index(drop=True, inplace=True)
    
    return (df)


def intertek_samples(df):
    ''' This function takes a clean df and extract samples'''
    
    # Find the indices of the start and end values, if they excist
    sample_ini = 0
    sample_end = df.index[df['SampleID']== 'CHECKS'][0]-2

    # Extract samples
    df_s = df.loc[sample_ini:sample_end]
    
    return df_s


def intertek_dup(df):
    ''' This function takes a clean df and extract duplicates'''
    
    # Find the indices of the start and end values, if they excist
    
    sample_end = df.index[df['SampleID']== 'CHECKS'][0]-2
    dup_ini = sample_end + 3
    dup_end = df.index[df['SampleID']== 'STANDARDS'][0]-2
    
    # Extract duplicates
    df_d = df.loc[dup_ini : dup_end]
    
    return df_d


# Function using pyplot

def duplicates(df, dup_list):
    ''' This function takes a clean df, and a list of samples. It returns a series of plots of duplicates'''
    
    plots = []  # List to store matplotlib figures
    report_data = [] # List to store report information
    # Filter df to include only samples of the list
    df_dup = df[df['SampleID'].isin(dup_list)]
    
    # Iterate through the list of duplicates and plot
    for sample in dup_list:
        df_i = df_dup[df_dup['SampleID']==sample].dropna(axis=1)
        
        x = df_i.iloc[[0], 2:]
        y = df_i.iloc[[1], 2:]
        
        original = x.values.flatten()
        duplicate = y.values.flatten()
        
        element_list = x.columns.tolist()
        
        # Plotting the data points
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(original, duplicate, color='blue', label='Duplicate vs Original')
        #ax.set_xlim(0, 25)
        
        # Set custom tick labels on x-axis
        ax.set_xticks(original)
        ax.set_xticklabels(element_list, rotation=45, ha='right', fontsize=8)    
        
        # Plotting the 1:1 line
        ax.plot(original, original, color='gray', linestyle='--', label='1:1 line')
    
        # Plotting the ±15% error threshold lines
        error_threshold = 0.15  # 15% error threshold
        upper_threshold = [(1 + error_threshold) * x for x in original]
        lower_threshold = [(1 - error_threshold) * x for x in original]
        ax.fill_between(original, lower_threshold, upper_threshold, interpolate=True, color='lightgray', alpha=0.3, label='15% Error Threshold')
    
        # Highlight dots outside the threshold in red
        outside_threshold = (duplicate > upper_threshold) | (duplicate < lower_threshold)
        ax.scatter(original[outside_threshold], duplicate[outside_threshold], color='red', label='Outside Threshold')
        
        
        # Adding labels and legend
        ax.set_xlabel('Original Measurements')
        ax.set_ylabel('Duplicate Measurements')
        ax.set_title(f'Scatter Plot of Duplicate vs Original Measurements for {sample}')
        
        # Add report
        elements_outside = [element_list[i] for i in range(len(element_list)) if outside_threshold[i]]
        report = f'Outside the ±15% error threshold: {elements_outside}'
        
        # Store report in report_data
        report_data.append((sample, elements_outside))
        
        ax.text(0.5, -0.25, report, horizontalalignment='center', verticalalignment='top', transform=ax.transAxes, fontsize=10, wrap=True)
        
        ax.legend()
    
        # Append the figure to the list
        plots.append(fig)
        
    # Convert report_data to a DataFrame
    report_df = pd.DataFrame(report_data, columns=['SampleID', 'Outside the ±15% error threshold'])
    
    
    
    return plots, report_df  # Return list of matplotlib figures and report data frame

def duplicates_px(df, dup_list):
    ''' This function takes a clean df, and a list of samples. It returns a series of plots of duplicates'''
    
    plots = []  # List to store Plotly figures
    report_data = []  # List to store report information
    
    # Filter df to include only samples in dup_list
    df_dup = df[df['SampleID'].isin(dup_list)]
    
    # Iterate through the list of duplicates and plot
    for sample in dup_list:
        df_i = df_dup[df_dup['SampleID'] == sample].dropna(axis=1)
        
        x = df_i.iloc[[0], 2:]
        y = df_i.iloc[[1], 2:]
        
        original = x.values.flatten()
        duplicate = y.values.flatten()
        
        element_list = x.columns.tolist()
        
        # Create a Plotly figure
        fig = go.Figure()
        
        # Scatter plot of original vs duplicate measurements
        fig.add_trace(go.Scatter(x=original, y=duplicate, mode='markers', marker=dict(color='blue'), name='Duplicate vs Original'))
        
        # 1:1 line
        fig.add_trace(go.Scatter(x=original, y=original, mode='lines', line=dict(color='gray', dash='dash'), name='1:1 line'))
        
        # ±15% error threshold lines
        error_threshold = 0.15  # 15% error threshold
        upper_threshold = [(1 + error_threshold) * x for x in original]
        lower_threshold = [(1 - error_threshold) * x for x in original]
        
        fig.add_trace(go.Scatter(x=original, y=upper_threshold, mode='lines', line=dict(color='lightgray', width=0), fill='tonexty', showlegend=False))
        fig.add_trace(go.Scatter(x=original, y=lower_threshold, mode='lines', line=dict(color='lightgray', width=0), fill='tonexty', showlegend=False))
        
        # Highlight dots outside the threshold in red
        outside_threshold = (duplicate > upper_threshold) | (duplicate < lower_threshold)
        fig.add_trace(go.Scatter(x=original[outside_threshold], y=duplicate[outside_threshold], mode='markers', marker=dict(color='red'), name='Outside Threshold'))
        
        # Layout customization
        fig.update_layout(title=f'Scatter Plot of Duplicate vs Original Measurements for {sample}',
                          xaxis_title='Original Measurements',
                          yaxis_title='Duplicate Measurements',
                          xaxis_tickangle=-45,
                          showlegend=True,
                          legend=dict(x=1.02, y=1.0, bgcolor='rgba(255, 255, 255, 0)', bordercolor='rgba(255, 255, 255, 0)'),
                          margin=dict(l=0, r=0, t=50, b=0),
                          hovermode='closest')
        # Update x-axis tick labels
        fig.update_xaxes(tickvals=original, ticktext=element_list)

        # Create report
        elements_outside = [element_list[i] for i in range(len(element_list)) if outside_threshold[i]]
        report = f'Outside the ±15% error threshold: {elements_outside}'
        
        # Store report in report_data
        report_data.append((sample, elements_outside))
        
        # Add report as annotation
        fig.add_annotation(
            x=0.5,
            y=-0.25,
            text=report,
            showarrow=False,
            font=dict(size=10),
            align='center',
            xref='paper',
            yref='paper'
        )
        
        # Append the figure to the list
        plots.append(fig)
        
    # Convert report_data to a DataFrame
    report_df = pd.DataFrame(report_data, columns=['SampleID', 'Outside the ±15% error threshold'])
    
    return plots, report_df  # Return list of Plotly figures and report data frame

def norm_REE (df, coeff_dict):
    # Make a copy of the df to avoid modifying the original data
    df_copy = df.copy()
    
    # list of REE from coeff_dict
    list_REE = list(coeff_dict.keys())
    df_REE = df_copy[list_REE] #Creates a subset of the data with just the REE
    # Iterate through the dictionary and apply the coefficients
    for column, coef in coeff_dict.items():
        if column in df_REE.columns:
            # Convert the coefficient to a float
            coef = float(coef[0])
            df_REE[column] /= coef  # Divide by the coefficient
    df_REE = pd.concat([df_copy.iloc[:,0], df_REE], axis = 1) # Get SampleID back
    df_REE.insert(0, 'ID', (range(1, len(df_REE) +1))) # Add a column with index values (for later plots)

    return df_REE

def norm_REE_df (df):
    # Make a copy of the df to avoid modifying the original data
    df_copy = df.copy()
    
    coeff_dict = {'La_ppm': {0: 0.245}, 'Ce_ppm': {0: 0.638}, 'Pr_ppm': {0: 0.0964}, 'Nd_ppm': {0: 0.474}, 'Sm_ppm': {0: 0.154}, 'Eu_ppm': {0: 0.058}, 'Gd_ppm': {0: 0.204}, 'Tb_ppm': {0: 0.0375}, 'Dy_ppm': {0: 0.254}, 'Ho_ppm': {0: 0.0567}, 'Er_ppm': {0: 0.166}, 'Tm_ppm': {0: 0.0256}, 'Yb_ppm': {0: 0.165}, 'Lu_ppm': {0: 0.0254}}
    
    # list of REE from coeff_dict
    list_REE = list(coeff_dict.keys())
    df_REE = df_copy[list_REE] #Creates a subset of the data with just the REE
    # Iterate through the dictionary and apply the coefficients
    for column, coef in coeff_dict.items():
        if column in df_REE.columns:
            # Convert the coefficient to a float
            coef = float(coef[0])
            df_REE[column] /= coef  # Divide by the coefficient
    df_REE = pd.concat([df_copy.iloc[:,0], df_REE], axis = 1) # Get SampleID back
    df_REE.insert(0, 'ID', (range(1, len(df_REE) +1))) # Add a column with index values (for later plots)

    return df_REE

def plot_REE (df_n, user_list):
    ''' Takes a normalised df and the list of samples to plot'''
    plots = []  # List to store matplotlib figures
    df_REE_l  = df_n.melt(id_vars = ["SampleID","ID"], var_name = 'Element', value_name = 'Value')
    #st.write(df_REE_l)
    for i in user_list:
        st.write("\n" , i)
        df_i = df_REE_l[df_REE_l['SampleID'] == i]  # Filter data for current SampleID
        
        
        # Create a new figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Use Seaborn to plot the line plot
        sns.lineplot(data=df_i, x='Element', y='Value', marker='o', markers=False, hue='ID', ax=ax)
        
        # Customize plot aesthetics
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.yscale('log')
        plt.ylabel('Chondrite normalized value')
        plt.xlabel('Element')
        plt.title(f'Chondrite-normalized REE diagram for SampleID: {i}')
        
        # Display the plot in Streamlit
        st.pyplot(fig)
        
        # Append the figure to the list (if you want to return the plots)
        plots.append(fig)

    return plots  # Return the list of figures if needed

def plot_REE_all (df_n):
    ''' Takes a normalised df and the list of samples to plot'''
    df_REE_l  = df_n.melt(id_vars = ["SampleID","ID"], var_name = 'Element', value_name = 'Value')
    #st.write(df_REE_l)
    
    # Create a new figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.lineplot(df_REE_l, x= "Element", y = 'Value',  hue = 'SampleID' , markers = True, legend = True)
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.yscale('log')
    plt.ylabel('Chondrite normalised value')
    plt.xlabel('Element')
    plt.title('Chondrite-normalised REE diagram')

    
    # Display the plot in Streamlit
    st.pyplot(fig)

def plot_oxides_line (df, user_list):
    ''' Takes a df (clean) and make boxplot of all samples chosen by user'''
    oxides = ['BaO_pct', 'CaO_pct', 'Cr2O3_pct','FeO_pct', 'Fe2O3_pct', 'K2O_pct', 'MgO_pct', 'MnO_pct', 'Na2O_pct', 'P2O5_pct', 'SiO2_pct' ]
    oxides_filter = ['SampleID'] + oxides
    
    df_o = df[oxides_filter]
    df_o = df_o.dropna(subset=df_o.columns[1:], how='all') # Drop empty rows
    df_o = df_o[df_o['SampleID'].isin(user_list)] #Only samples that are in user list
    
    df_o_l  = df_o.melt(id_vars = ["SampleID"], var_name = 'Element', value_name = 'Value')
    
    for sample in user_list:
        df_i = df_o_l[df_o_l['SampleID']== sample]
        # Create a new figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax = sns.lineplot(df_i, x= "Element", y = 'Value')#, hue = 'SampleID')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.yscale('log')
        plt.ylabel('[%]')
        plt.xlabel('Element')
        plt.title(f'Major oxides composition for {sample}')
        
        # Annotate with counts from df_h
        counts = df_i[df_i["Element"].isin(oxides)].groupby("Element").size()
        
        ax.text(9, 7, ('n= ' + str(counts[0])), ha='center', color='black')
        
        # Display the plot in Streamlit
        st.pyplot(fig)
    
    st.write ("This is the NTGS Standard major Oxide composition")
    st.write(df_o)
    
def plot_blanks(df):
    ''' This function takes a df0 and plots blacks'''
    
    df_blanks = df[df['ELEMENTS'] == 'Control Blank']
    
    # Convert numeric columns to numeric data type (excluding the 'ELEMENTS' column)
    #df_blanks.iloc[:, 1:] = df_blanks.iloc[:, 1:].apply(pd.to_numeric)
    
    # Replace NaN for 0
    df_blanks = df_blanks.fillna(0)
    
    # Melt the DataFrame to long format for plotting
    df_melted = df_blanks.melt(id_vars='ELEMENTS', var_name='element', value_name='Values', ignore_index = False)
    
    # Convert values to numeric
    df_melted.iloc[:,2] = df_melted.iloc[:,2].apply(pd.to_numeric)
    
    # Plot using seaborn
    
    # Create a new figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.lineplot(data=df_melted, x='element', y='Values',hue= df_melted.index, marker='o', markersize=5)
    
    # Set title and labels
    plt.title('Line Plot of blanks for Elements')
    plt.xlabel('Elements')
    plt.ylabel('Values')
    plt.xticks(rotation=90, fontsize = 5)
    plt.ylim(-1, 1)
    plt.grid(False)
    
    plt.legend(title='Blanks', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Display the plot in Streamlit
    st.pyplot(fig)
    
def df_blanks (df):
    ''' This formula takes a df0 and reports on 'Control Blank' results' of new batch'''
    # No of 'Control Blank' analyses
    noBlank = df['ELEMENTS'].value_counts()['Control Blank']
    st.write('This batch contains', str(noBlank), 'control blanks')
    
    blanks = df[df['ELEMENTS'] == 'Control Blank']
    
    # Convert numeric columns to numeric data type (excluding the 'ELEMENTS' column)
    blanks.iloc[:, 1:] = blanks.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
    
    # Create a boolean mask for null and NaN values
    null_mask = blanks.iloc[:, 1:].isnull()

    # Exclude null and NaN values before checking for positive values
    non_null_blanks = blanks.iloc[:, 1:][~null_mask]

    # Check for positive values in the numeric columns
    blank_positive = (non_null_blanks > 0).any()

    # Find columns with positive values
    positive_columns = blank_positive[blank_positive].index.tolist()
 
    if len(positive_columns) == 0:
        st.write("All values are below detection limit (negative)")
    else:
        st.write("The following elements returned values over the detection limit (positive): ", positive_columns, "Please check")

def plot_blanks_px(df):
    ''' This function takes a df and plots blanks'''
    
    df_blanks = df[df['ELEMENTS'] == 'Control Blank']
    
    # Replace NaN with 0
    df_blanks = df_blanks.fillna(0)
    
    # Melt the DataFrame to long format for plotting
    df_melted = df_blanks.melt(id_vars='ELEMENTS', var_name='element', value_name='Values')
    
    # Convert values to numeric
    df_melted['Values'] = pd.to_numeric(df_melted['Values'])
    
    # Plot using Plotly Express
    fig = px.line(df_melted, x='element', y='Values', title='Line Plot of Blanks for Elements', color='ELEMENTS',
                  line_shape='linear', render_mode='svg', labels={'element': 'Elements', 'Values': 'Values'})
    
    # Customize layout
    fig.update_layout(xaxis_tickangle=-45, yaxis_range=[-1, 1], legend_title='Blanks',
                      legend=dict(x=1.02, y=1.0, bgcolor='rgba(255, 255, 255, 0)', bordercolor='rgba(255, 255, 255, 0)'),
                      margin=dict(l=0, r=0, t=50, b=0))
    
    # Display the plot in Streamlit
    st.plotly_chart(fig)
   

    
NTGS_std_list = ['AS08JAW004','AS08JAW006', 'DW06LMG005']

# Main Streamlit function
def main():
    
    
    st.title("NTGS - Amazing QA/QC app 0.1")

    # Sidebar - File upload and filtering options
    st.sidebar.title("Upload CSV file")
    uploaded_file = st.sidebar.file_uploader("Upload Intertek CSV file", type=['csv'])
    

    if uploaded_file is not None:
        # Load data into DataFrame
        df0 = load_data(uploaded_file)
        
        new_batch = uploaded_file.name
        st.write(f'### Batch: {new_batch}')
        
        st.write(df0.head(6))
        st.write('Dios mio!! What an ugly table - Lets clean it a bit')
        
        st.sidebar.title ("Step 1- Clean data")
        if st.sidebar.checkbox('Clean data'):
            st.write ("Much better...")
            df = intertek_clean(df0)
            st.write(df.head(6))
            
            df_d = intertek_dup(df)
            
            if st.checkbox('Export duplicated samples (only) to CSV'):
                df_s = intertek_dup(df)
                csv = df_s.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()  # B64 encoding
                href = f'<a href="data:file/csv;base64,{b64}" download="Duplicated_samples_{new_batch}">Download CSV File</a>'
                st.markdown(href, unsafe_allow_html=True)
                
            st.write("The clean dataset is ready to be imported into IoGas - Export the data and open it in IoGas")
            
            if st.checkbox('Export samples to CSV'):
                df_s = intertek_samples(df)
                csv = df_s.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()  # B64 encoding
                href = f'<a href="data:file/csv;base64,{b64}" download="Sample_data_{new_batch}">Download CSV File</a>'
                st.markdown(href, unsafe_allow_html=True)
        
            st.sidebar.title ("Step 2- Duplicates")
            
            dup_list = st.sidebar.multiselect('Select duplicates', df_d['SampleID'])
            
            if st.sidebar.checkbox('Plot of selected duplicates and report table', key = 'dup_plots'):
                plots, report_df = duplicates_px(df, dup_list)  # Calling duplicates() to get plots
                st.header('Plots of Duplicate Analysis')
                
                for plot in plots:
                    #st.pyplot(plot)  # Display each plot using st.pyplot()
                    st.plotly_chart(plot)
                st.write ("### Report")
                st.dataframe(report_df)
                
                if st.checkbox('Export report as csv'):
                    # Export report_df to CSV and provide download link
                    csv = report_df.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()  # Base64 encoding for download link
                    href = f'<a href="data:file/csv;base64,{b64}" download="Duplicates_report_{new_batch}">Download CSV File</a>'
                    st.markdown(href, unsafe_allow_html=True)
            st.write ("Note to myself: report method associated with bad duplicates, and maybe quantify the % error")
            
            st.sidebar.title ("Step 3 - NTGS Standards REE")
            
            df_s = intertek_samples(df)
            options = df_s[df_s['SampleID'].isin(NTGS_std_list)]['SampleID'].tolist()

            if options is not None:
                NTGS_std = st.sidebar.multiselect('Select the NTGS Standards', options = options)
                st.sidebar.title("Upload REE reference table or choose internal reference")
                uploaded_REE = st.sidebar.file_uploader("Upload Normalising REE table as CSV file", type=['csv'])
                
                
                if uploaded_REE is not None:
                    REE_coeff = load_data(uploaded_REE).to_dict() # creates a coefficient dictionary from REE df
                    st.write('#### These are the REE normalizing coefficients')
                    st.write(pd.DataFrame(REE_coeff).head())
                    st.sidebar.title('Plot normalised REE patterns')
                    df_REE = df_s[df_s['SampleID'].isin(options)]
                    df_REE_n = norm_REE(df_REE, REE_coeff)
                    
                    
                    if st.sidebar.checkbox('Plot REE from samples in this batch'):
                        st.write('#### This is the normalised data')                   
                        st.write(df_REE_n)
                        plots = plot_REE(df_REE_n, NTGS_std)
                        
                    if st.sidebar.checkbox('Plot REE _ all in one plot'): 
                        st.write('#### This is the normalised data')
                        st.write(df_REE_n)
                        plot_REE_all(df_REE_n, NTGS_std)
            else:
                st.write ('There are no NTGS Standards')
                    
                if st.sidebar.checkbox("You do not have a reference? No worries, just use our internal reference (reference)"):
                    st.sidebar.title('Plot normalised REE patterns')
                    df_REE = df_s[df_s['SampleID'].isin(options)]
                    df_REE_i = norm_REE_df(df_REE)
                    
                    if st.sidebar.checkbox('Plot REE from samples in this batch'):
                        st.write('#### This is the normalised data')                   
                        st.write(df_REE_i)
                        plots = plot_REE(df_REE_i, NTGS_std)
                        
                    if st.sidebar.checkbox('Plot REE _ all in one plot'): 
                        st.write('#### This is the normalised data')
                        st.write(df_REE_i)
                        plot_REE_all(df_REE_i)
                
            st.sidebar.title ("Step 3a - NTGS Standards Major Oxides")
            
            if st.sidebar.checkbox ("Plot Major Oxides of chosen samples"):
                st.write("### This is the NTGS Standards major oxides composition")
                plot_oxides_line(df_s, NTGS_std)
                
            st.sidebar.title ("Step 4 - Blanks")
            if st.sidebar.checkbox('Show me the blanks and report'):
                st.write("### These are the blanks in your batch")
                plot_blanks_px(df0)
                df_blanks(df0)
    
        
# Run the Streamlit app
if __name__ == "__main__":
    main()
