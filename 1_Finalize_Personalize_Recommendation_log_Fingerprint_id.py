import pandas as pd
from collections import Counter
from scipy.stats import pearsonr

def clean_data(data):
    # Remove rows with NaN values in either 'post_id' or 'finger_print_id'
    data = data.dropna(subset=['post_id', 'finger_print_id'])
    return data

def calculate_similarity(input_data, data):
    interaction_matrix = data.pivot_table(index='finger_print_id', columns='post_id', aggfunc='size', fill_value=0)
    
    similar_users = {}

    for input_finger_print_id in input_data['finger_print_id'].unique():
        input_user_vector = interaction_matrix.loc[input_finger_print_id]

        similar_users[input_finger_print_id] = []
        
        for finger_print_id in interaction_matrix.index:
            if finger_print_id != input_finger_print_id:
                finger_print_vector = interaction_matrix.loc[finger_print_id]
                if len(input_user_vector) > 1 and len(finger_print_vector) > 1:
                    correlation, _ = pearsonr(input_user_vector, finger_print_vector)
                    if not pd.isna(correlation):
                        similar_users[input_finger_print_id].append((finger_print_id, correlation))
    return similar_users

def find_top_similar_user(similar_users, top_n):
    top_similar_users = {}
    for input_finger_print_id, user_similarities in similar_users.items():
        user_similarities.sort(key=lambda x: x[1], reverse=True)
        top_similar_users[input_finger_print_id] = user_similarities[:top_n]
    return top_similar_users

def recommend_posts(input_data, top_similar_users, data, top_n=12, min_recommendations=12):
    recommendations = {}
    post_frequency = Counter(data['post_id'].dropna())
    for input_finger_print_id, similar_user_list in top_similar_users.items():
        recommendations[input_finger_print_id] = []
        input_user_posts = set(input_data[input_data['finger_print_id'] == input_finger_print_id]['post_id'].dropna())
        additional_similar_users = []
        combined_similar_users = similar_user_list + additional_similar_users
        for finger_print_id, similarity_score in combined_similar_users:
            user_posts = set(data[data['finger_print_id'] == finger_print_id]['post_id'].dropna())
            recommended_posts = user_posts - input_user_posts
            recommendations[input_finger_print_id].extend(recommended_posts)
            if len(recommendations[input_finger_print_id]) >= min_recommendations:
                break
        if len(recommendations[input_finger_print_id]) < min_recommendations:
            frequent_posts = [post for post, _ in post_frequency.most_common() if post not in input_user_posts]
            recommendations[input_finger_print_id].extend(frequent_posts[:min_recommendations - len(recommendations[input_finger_print_id])])
        recommendations[input_finger_print_id] = [int(post) for post in recommendations[input_finger_print_id] if post != -1]
    return recommendations

def get_personalized_recommendations(finger_print_id, data, top_n=12, min_recommendations=12):
    user_posts = data[data['finger_print_id'] == finger_print_id]['post_id'].dropna().tolist()
    
    if not user_posts:
        print(f"finger_print_id {finger_print_id} has no posts.")
        return

    input_data = pd.DataFrame({
        'finger_print_id': [finger_print_id] * len(user_posts),
        'post_id': user_posts
    })

    similarities = calculate_similarity(input_data, data)
    top_similar_users = find_top_similar_user(similarities, top_n)
    recommendations = recommend_posts(input_data, top_similar_users, data, top_n, min_recommendations)

    # Print results
    print("------------------Top similar users-----------------------")
    for input_finger_print_id, similar_user_list in top_similar_users.items():
        print(f"finger_print_id {input_finger_print_id} has top similar users:")
        for finger_print_id, similarity_score in similar_user_list:
            print(f"  finger_print_id {finger_print_id} with similarity score: {similarity_score}")


    
    # Additional print statements
    print("------------------Matching similar personalize-----------------------")
    for input_finger_print_id, similar_user_list in top_similar_users.items():
        for finger_print_id, similarity_score in similar_user_list:
            input_user_posts = set(map(int, input_data[input_data['finger_print_id'] == input_finger_print_id]['post_id']))
            matching_user_posts = set(map(int, data[data['finger_print_id'] == finger_print_id]['post_id']))
            print(f"Input finger_print_id {input_finger_print_id} has post_ids: {input_user_posts}")
            print(f"Matching finger_print_id {finger_print_id} has post_ids: {matching_user_posts}")
            print("\n")
            
            
    print("\n------------------Recommendations-----------------------")
    for input_finger_print_id, recommended_posts in recommendations.items():
        print(f"finger_print_id {input_finger_print_id} should recommend the following posts:")
        print(recommended_posts)
            
            

    print("Algorithm User Based ------>  Recommendations personalize:\n")
    recommended_posts_listed = []

    for input_finger_print_id, recommended_posts in recommendations.items():
        print(f"finger_print_id {input_finger_print_id} should recommend the following posts:")
        print(recommended_posts)
        recommended_posts_listed.extend(recommended_posts)

    print("\nRecommended posts All listed:", recommended_posts_listed)

    # Dictionary to store matching user's post_ids
    matching_post_ids = {}

    # Collect matching user's post_ids
    for input_finger_print_id, similar_user_list in top_similar_users.items():
        for finger_print_id, similarity_score in similar_user_list:
            input_user_post_ids = set(input_data[input_data['finger_print_id'] == input_finger_print_id]['post_id'])
            matching_user_post_ids = set(data[data['finger_print_id'] == finger_print_id]['post_id'])
            matching_post_ids[input_finger_print_id] = input_user_post_ids.intersection(matching_user_post_ids)

    # Check if there are any matching post IDs
    all_matching_post_ids = [post_id for post_ids in matching_post_ids.values() for post_id in post_ids]
    if all_matching_post_ids:
        most_common_post_id = Counter(all_matching_post_ids).most_common(1)[0][0]
        print(f"\nMost common post_id among matching users: {most_common_post_id}")
    else:
        print("\nNo common post_id found among matching users.")

# Example usage:
data = pd.read_csv('blog_viewerpreference_202406210935.csv')

# Clean data
data = clean_data(data)

# Ensure that NaN values are not present by dropping rows with -1 values
data = data[(data['post_id'] != -1) & (data['topic_id'] != -1)]

input_finger_print_id = input("Enter the finger_print_id you want to get recommendations for: ")
get_personalized_recommendations(input_finger_print_id, data)
