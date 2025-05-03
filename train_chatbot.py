    import json
    import numpy as np
    import nltk
    import random
    import pickle
    from nltk.stem import WordNetLemmatizer
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.optimizers import SGD

    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Load intents from train_data.json
    with open("train_data.json", encoding="utf-8") as file:
        intents = json.load(file)

    words = []
    classes = []
    documents = []
    ignore_letters = ["?", "!", ".", ","]

    # Preprocess the training data
    for intent in intents["intents"]:
        for pattern in intent["patterns"]:
            word_list = nltk.word_tokenize(pattern)
            words.extend(word_list)
            documents.append((word_list, intent["tag"]))
            if intent["tag"] not in classes:
                classes.append(intent["tag"])

    words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
    words = sorted(set(words))
    classes = sorted(set(classes))

    # Save words and classes for later use
    pickle.dump(words, open("words.pkl", "wb"))
    pickle.dump(classes, open("classes.pkl", "wb"))

    # Create training data
    training = []
    output_empty = [0] * len(classes)

    for document in documents:
        bag = []
        word_patterns = document[0]
        word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
        
        for word in words:
            bag.append(1 if word in word_patterns else 0)

        output_row = list(output_empty)
        output_row[classes.index(document[1])] = 1
        training.append([bag, output_row])

    # Convert to NumPy arrays
    random.shuffle(training)
    training = np.array(training, dtype=object)
    train_x = np.array(list(training[:, 0]))
    train_y = np.array(list(training[:, 1]))

    # Define model architecture
    model = Sequential([
        Dense(128, input_shape=(len(train_x[0]),), activation="relu"),
        Dropout(0.5),
        Dense(64, activation="relu"),
        Dropout(0.5),
        Dense(len(classes), activation="softmax")
    ])

    # Compile model
    sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
    model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

    # Train model
    model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

    # Save model
    model.save("chatbot_model.h5")
    print("Model training complete.")
