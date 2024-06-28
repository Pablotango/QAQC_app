import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import base64  # Import the base64 module

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
    
    # Delete first rows (metadata)
    
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


# Main Streamlit function
def main():
    st.title("NTGS - Amazing QA/QC app")

    # Sidebar - File upload and filtering options
    st.sidebar.title("Upload CSV file")
    uploaded_file = st.sidebar.file_uploader("Upload Intertek CSV file", type=['csv'])
    

    if uploaded_file is not None:
        # Load data into DataFrame
        df0 = load_data(uploaded_file)
        
        new_batch = uploaded_file.name
        st.write(f'Batch name {new_batch}')
        
        st.write(df0.head(6))
        st.write("Dios mio!! What an uggly table - Lets clean it a bit")
        
        if st.checkbox('Clean data'):
            st.write ("Much better...")
            df = intertek_clean(df0)
            st.write(df.head(6))
            
            df_d = intertek_dup(df)
            dup_list = st.sidebar.multiselect('Select duplicates', df_d['SampleID'])
            
            if st.checkbox('Export clean duplicated samples to CSV'):
                df_s = intertek_dup(df)
                csv = df_s.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()  # B64 encoding
                href = f'<a href="data:file/csv;base64,{b64}" download="Duplicated_samples_{new_batch}">Download CSV File</a>'
                st.markdown(href, unsafe_allow_html=True)
                
            if st.checkbox('Export clean samples to CSV'):
                df_s = intertek_samples(df)
                csv = df_s.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()  # B64 encoding
                href = f'<a href="data:file/csv;base64,{b64}" download="Sample_data_{new_batch}">Download CSV File</a>'
                st.markdown(href, unsafe_allow_html=True)
                
            if st.checkbox('Choose the dupliccates (sidebar) and plot'):
                plots, report_df = duplicates(df, dup_list)  # Calling duplicates() to get plots
                st.header('Plots of Duplicate Analysis')
                
                for plot in plots:
                    st.pyplot(plot)  # Display each plot using st.pyplot()
                st.write ("# Report")
                st.dataframe(report_df)
                if st.checkbox('Export report as csv'):
                    # Export report_df to CSV and provide download link
                    csv = report_df.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()  # Base64 encoding for download link
                    href = f'<a href="data:file/csv;base64,{b64}" download="Duplicates_report_{new_batch}">Download CSV File</a>'
                    st.markdown(href, unsafe_allow_html=True)
            
# Run the Streamlit app
if __name__ == "__main__":
    main()
