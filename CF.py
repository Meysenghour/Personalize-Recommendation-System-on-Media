from collections import defaultdict

def calculate_similarity(input_data, data):
    similar_users = defaultdict(list)
    similar_items = defaultdict(list)
    user_posts = data.groupby('user_id')['post_id'].apply(set).to_dict()

    for input_user_id in input_data['user_id'].unique():
        input_user_posts = set(input_data[input_data['user_id'] == input_user_id]['post_id'])

        for user_id, posts in user_posts.items():
            common_posts = input_user_posts.intersection(posts)
            if common_posts:
                similarity_score = len(common_posts) / len(input_user_posts.union(posts))
                similar_users[input_user_id].append((user_id, similarity_score))
                for item in common_posts:
                    similar_items[item].append((user_id, similarity_score))

    return similar_users, similar_items

def find_top_similar_user(similar_users, top_n=1):
    top_similar_users = {}

    for input_user_id, user_similarities in similar_users.items():
        user_similarities.sort(key=lambda x: x[1], reverse=True)
        top_similar_users[input_user_id] = user_similarities[:top_n]

    return top_similar_users

def recommend_posts(input_data, similar_users, similar_items, data):
    recommendations = defaultdict(list)

    for input_user_id, similar_user_list in similar_users.items():
        for user_id, _ in similar_user_list:
            user_posts = set(data[data['user_id'] == user_id]['post_id'])
            input_user_posts = set(input_data[input_data['user_id'] == input_user_id]['post_id'])
            recommended_posts = user_posts - input_user_posts
            recommendations[input_user_id].extend(recommended_posts)

    for input_user_id, similar_item_list in similar_items.items():
        for user_id, _ in similar_item_list:
            user_posts = set(data[data['user_id'] == user_id]['post_id'])
            input_user_posts = set(input_data[input_data['user_id'] == input_user_id]['post_id'])
            recommended_posts = user_posts - input_user_posts
            recommendations[input_user_id].extend(recommended_posts)

    return recommendations

similar_users, similar_items = calculate_similarity(input_data, data)
top_similar_users = find_top_similar_user(similar_users, top_n=1)
recommendations = recommend_posts(input_data, top_similar_users, similar_items, data)

print("Top similar users:")
for input_user_id, similar_user_list in top_similar_users.items():
    print(f"user_id {input_user_id} has top similar to:")
    for user_id, similarity_score in similar_user_list:
        print(f"  user_id {user_id} with similarity score: {similarity_score}")

print("\nRecommendations:")
for input_user_id, recommended_posts in recommendations.items():
    print(f"user_id {input_user_id} should recommend the following posts:")
    print(recommended_posts)
