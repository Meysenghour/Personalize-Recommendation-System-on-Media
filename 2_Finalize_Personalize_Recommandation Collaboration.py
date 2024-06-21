import pandas as pd
from collections import Counter
import pandas as pd
from scipy.stats import pearsonr


def calculate_similarity(input_data, data):
    # Combine input_data and data to ensure all users and posts are included
    combined_data = pd.concat([input_data, data])
    
    # Create a user-post interaction matrix
    interaction_matrix = combined_data.pivot_table(index='user_id', columns='post_id', aggfunc='size', fill_value=0)
    
    similar_users = {}

    for input_user_id in input_data['user_id'].unique():
        input_user_vector = interaction_matrix.loc[input_user_id]

        similar_users[input_user_id] = []
        
        for user_id in interaction_matrix.index:
            if user_id != input_user_id:
                user_vector = interaction_matrix.loc[user_id]
                
                # Calculate Pearson Correlation Coefficient
                if len(input_user_vector) > 1 and len(user_vector) > 1:
                    correlation, _ = pearsonr(input_user_vector, user_vector)
                    if not pd.isna(correlation):  # Ensure the correlation is not NaN
                        similar_users[input_user_id].append((user_id, correlation))
    
    return similar_users


def find_top_similar_user(similar_users, top_n):
    top_similar_users = {}

    for input_user_id, user_similarities in similar_users.items():
        user_similarities.sort(key=lambda x: x[1], reverse=True)
        top_similar_users[input_user_id] = user_similarities[:top_n]

    return top_similar_users

def recommend_posts(input_data, similar_users, data, top_n=1, min_recommendations=5):
    recommendations = {}
    
    # Counting the frequency of each post in the dataset
    post_frequency = Counter(data['post_id'])

    for input_user_id, similar_user_list in similar_users.items():
        recommendations[input_user_id] = []
        
        # Posts already seen by the input user
        input_user_posts = set(input_data[input_data['user_id'] == input_user_id]['post_id'])
        
        # Extend the similar user list if necessary
        additional_similar_users = []
        if len(similar_user_list) < top_n:
            additional_similar_users = similar_users[input_user_id][top_n:top_n*2]

        combined_similar_users = similar_user_list + additional_similar_users
        
        for user_id, similarity_score in combined_similar_users:
            user_posts = set(data[data['user_id'] == user_id]['post_id'])
            recommended_posts = user_posts - input_user_posts
            recommendations[input_user_id].extend(recommended_posts)
            
            if len(recommendations[input_user_id]) >= min_recommendations:
                break
        
        # If there are still not enough recommendations, add the most frequent posts
        if len(recommendations[input_user_id]) < min_recommendations:
            frequent_posts = [post for post, _ in post_frequency.most_common() if post not in input_user_posts]
            recommendations[input_user_id].extend(frequent_posts[:min_recommendations - len(recommendations[input_user_id])])
    
    return recommendations

# Example usage:
# input_data, similar_users, and data should be defined as per your specific dataset and context
similarities = calculate_similarity(input_data, data)
top_similar_users = find_top_similar_user(similarities, top_n=1)
recommendations = recommend_posts(input_data, top_similar_users, data)

print("------------------Top similar users-----------------------\n")
for input_user_id, similar_user_list in top_similar_users.items():
    print(f"user_id {input_user_id} has top similar to:")
    for user_id, similarity_score in similar_user_list:
        print(f"  user_id {user_id} with similarity score: {similarity_score},\n")
# Additional print statements
print("------------------Matching similar personalize-----------------------")
for input_user_id, similar_user_list in top_similar_users.items():
    for user_id, similarity_score in similar_user_list:
        print(f"Input user_id {input_user_id} has post_ids: {set(input_data[input_data['user_id'] == input_user_id]['post_id'])}")
        print(f"Matching user_id {user_id} has post_ids: {set(data[data['user_id'] == user_id]['post_id'])}")
        print("\n")
print("Algorithm User Based ------>  Recommendations personalize:\n")
recommended_posts_listed = []

for input_user_id, recommended_posts in recommendations.items():
    print(f"user_id {input_user_id} should recommend the following posts:")
    print(recommended_posts)
    recommended_posts_listed.extend(recommended_posts)

print("\nRecommended posts All listed:", recommended_posts_listed)
from collections import Counter   
# Dictionary to store matching user's post_ids
matching_post_ids = {}

# Collect matching user's post_ids
for input_user_id, similar_user_list in top_similar_users.items():
    for user_id, similarity_score in similar_user_list:
        input_user_post_ids = set(input_data[input_data['user_id'] == input_user_id]['post_id'])
        matching_user_post_ids = set(data[data['user_id'] == user_id]['post_id'])
        matching_post_ids[input_user_id] = input_user_post_ids.intersection(matching_user_post_ids)

# Find the post_id with the most users among matching users
most_common_post_id = Counter([post_id for post_ids in matching_post_ids.values() for post_id in post_ids]).most_common(1)[0][0]


