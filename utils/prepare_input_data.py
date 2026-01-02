import pandas as pd
import os
import requests

TEXT_TEMPLATE="""
Video title: {title}

Video description: {video_description}

After analise the metadata and the thumbnail
Answer 'yes' if the video is clearly not user generated content related to sports and not recorded from the stands 
else answer 'no' 
"""

SYSTEM_ROLE="You are a helpful assistant searching for user generated videos related to sport events from youtube."

def prepare_message(title, video_description, thumbnail):
    text_prompt = TEXT_TEMPLATE.format(title = title, video_description = video_description)
    message = [
        {"role": "system", "content": SYSTEM_ROLE},
        {"role": "user", "content": [{"type": "image", "image": thumbnail}, {"type": "text", "text": text_prompt},],}
    ]
    return message

def prepare_batch(csv_path, num_batch, limit = None):
    """
    Returns a list of messages for batch processing in qwen 2.5vl
    
    :param csv_path: Description
    :param num_batch: Description
    """

    df = pd.read_csv(csv_path, sep=',')
    if limit:
        df = df.head(limit)

    filtered_data = df[['title','video_description', 'thumbnail']].to_numpy().tolist()
    all_messages = []
    for data in filtered_data:
        new_message = prepare_message(data[0], data[1], data[2])
        all_messages.append(new_message)
    
    total_len = len(all_messages)
    batched_messages = [all_messages[x:x+num_batch] for x in range(0, total_len, num_batch)]

    return batched_messages



    



