from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

ARTICLE = """ Known for his trademark gelled spiked up crew cut with sunglasses, Shah Rukh Khan (SRK) is an Indian Bollywood movie star, movie producer, magazine model, showman, public speaker, author, philanthropist and television host/personality working predominantly in Hindi cinema.

Khan began his on-camera acting debut in 1987 at the age of 21 by guest starring in various Indian serial drama soap opera TV shows as well as appearing in numerous television commercials and brand advertisements for products. He studied theatre arts and drama during his second year of college after participating in numerous school plays. He landed a few acting gigs in Delhi, before acting on Indian TV and became popular. Khan came down to Mumbai to shoot a TV series with self doubt and under confidence. So he came for a year to give it a shot. He began auditioning for starring roles in Hindi movies in 1990 after the death of his mother (his father died a decade earlier in 1980). Khan's parents died early, which made him heartbroken in Delhi. Khan decided to pursue a full-time acting career and relocating to Mumbai to start afresh, hoping to enjoy acting and overcome the dejecting death of his parents, as there was nothing for him to go back to. He began auditioning for starring roles in Hindi movies in 1990 after the death of his mother (his father died a decade earlier in 1980).

After recuperating from a career-ending sports injury, he landed his breakout breakthrough feature film starring role in the Silver screen in June 1992, and rose to prominence in the mid-to-late 1990s. Khan shot to stardom in his first feature film "Deewana" (1992) which won him the first of 13 Filmfare awards -- the Bollywood equivalent of an Oscar. He continued starring in blockbuster movies throughout the 2000s with a mixed bag of career fluctuations, establishing himself as a very bankable, versatile movie star in the early-to-mid 2010s. Following a 4-year sabbatical hiatus in the wake of the corona-virus pandemic, Khan made a resurgence comeback in 2023, and continues to act and star in A-Lister blockbuster movies.

Referred to in the media as the "Baadshah of Bollywood" and "King Khan", he has appeared in over 100 films, and earned numerous accolades, including 14 Filmfare Awards.

He has been awarded the Padma Shri by the Government of India, as well as the Ordre des Arts et des Lettres and Legion of Honour by the Government of France. Khan has a significant following in Asia and the Indian diaspora worldwide. In terms of audience size and income, several media outlets have described him as one of the most successful film stars in the world. Many of his films depict and portray Indian national identity, Indian patriotism, and connections with diaspora communities, or gender, racial, social and religious differences and grievances.

Shah Rukh Khan has been TAG Heuer's brand ambassador in India since September 2003 & is close friends with Amir. He is Bollywood's most bankable movie stars with brand endorsements & resides in the affluent suburbs of Bandra, Mumbai, India with his wife and children.
"""
print(summarizer(ARTICLE, max_length=130, min_length=30, do_sample=False))
