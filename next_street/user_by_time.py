"""
script to analyze users by time
"""


import pandas as pd
import matplotlib.pyplot as plt


def main():
    """
    run the graph by time
    """
    user_df = pd.read_csv("data/user_08042023.csv")
    user_df = user_df.sort_values(by='createdAt', ascending=False)
    print(user_df.head(10))

    # Convert the 'createdAt' column to pandas datetime format
    user_df['createdAt'] = pd.to_datetime(user_df['createdAt'])

    # Group the DataFrame by day and count the total number of users per day
    users_by_day = user_df.groupby(user_df['createdAt'].dt.date).size()

    # Create the line graph
    plt.figure(figsize=(10, 6))
    plt.plot(users_by_day.index, users_by_day.values, marker='o', linestyle='-', color='b')
    plt.xlabel('Date')
    plt.ylabel('Total Users')
    plt.title('Total Users by Day')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("pngs/users_by_day.png")
    plt.close()

    # Calculate the cumulative sum of users by day
    cumulative_users_by_day = users_by_day.cumsum()

    # Create the cumulative line graph
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_users_by_day.index, cumulative_users_by_day.values, marker='o', linestyle='-', color='b')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Users')
    plt.title('Cumulative Users by Day')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("pngs/cumulative_users_by_day.png")
    plt.close()




if __name__ == "__main__":
    main()