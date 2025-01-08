print("********** WELCOME TO THE QUIZ GAME *************")
ans = input("Are You Ready For the Quiz? (Yes/No): ").strip().upper()

if ans == "YES":
    print("OK..! Let's Begin.")
    score = 0
    question_bank = [
        {"text": "1. What is 2 + 7?", "answer": "B"},
        {"text": "2. Output of the code: print(2 + 2 * 3)?", "answer": "A"},
        {"text": "3. What is 10 + 20?", "answer": "D"},
        {"text": "4. Which keyword is used to define a function in Python?", "answer": "A"},
        {"text": "5. Which of the following data types is mutable in Python?", "answer": "D"},
        {"text": "6. What is the capital of Australia?", "answer": "C"},
        {"text": "7. What is the largest ocean in the world?", "answer": "D"},
        {"text": "8. Who wrote the play Romeo and Juliet?", "answer": "A"},
        {"text": "9. What is the currency of Japan?", "answer": "A"},
        {"text": "10. What is the tallest mountain in the world?", "answer": "B"},
        {"text": "11. What is the largest organ in the human body?", "answer": "D"},
        {"text": "12. Which is the correct way to comment multiple lines in Python?", "answer": "C"},
        {"text": "13. What is the output of the following code: print('Hello' + 3)?", "answer": "C"},
        {"text": "14. Which is used to read user input in Python 3?", "answer": "A"},
        {"text": "15. Who is the author of 'To Kill a Mockingbird'?", "answer": "B"}
    ]
    options = [
        ["A. 3", "B. 9", "C. 6", "D. 10"],
        ["A. 8", "B. 10", "C. 6", "D. 12"],
        ["A. 23", "B. 34", "C. 28", "D. 30"],
        ["A. def", "B. function", "C. define", "D. func"],
        ["A. int", "B. str", "C. tuple", "D. list"],
        ["A. Sydney", "B. Melbourne", "C. Canberra", "D. Perth"],
        ["A. Atlantic Ocean", "B. Indian Ocean", "C. Arctic Ocean", "D. Pacific Ocean"],
        ["A. William Shakespeare", "B. Jane Austen", "C. Charles Dickens", "D. F. Scott Fitzgerald"],
        ["A. Yen", "B. Euro", "C. Dollar", "D. Rupee"],
        ["A. Mount Kilimanjaro", "B. Mount Everest", "C. Mount McKinley", "D. Mount Fuji"],
        ["A. Heart", "B. Liver", "C. Brain", "D. Skin"],
        ["A. /* This is a comment */", "B. # This is a comment #", "C. ''' This is a comment '''", "D. // This is a comment //"],
        ["A. Hello3", "B. HelloHello", "C. Error", "D. 3Hello"],
        ["A. input()", "B. raw_input()", "C. get_input()", "D. read_input()"],
        ["A. J.K. Rowling", "B. Harper Lee", "C. Ernest Hemingway", "D. F. Scott Fitzgerald"]
    ]

    def check_answer(user_guess, correct_answer):
        return user_guess == correct_answer

    for question_num in range(len(question_bank)):
        print("--------------------------------------")
        print(question_bank[question_num]["text"])
        for i in options[question_num]:
            print(i)

        while True:
            guess = input("Enter Your Answer (A/B/C/D) or type 'Q' to quit: ").strip().upper()
            if guess in ["A", "B", "C", "D", "Q"]:
                break
            else:
                print("Invalid input. Please enter A, B, C, D, or Q.")

        if guess == "Q":
            print("You chose to quit the quiz.")
            break

        is_correct = check_answer(guess, question_bank[question_num]["answer"])
        if is_correct:
            print("Correct Answer!")
            score += 1
        else:
            print("Wrong Answer!")
            print(f"The Correct Answer is: {question_bank[question_num]['answer']}")
        print(f"Your current score is: {score}/{question_num + 1}")

    print(f"\nYou have given {score} correct answers.")
    print(f"Your Score is {(score / len(question_bank)) * 100:.2f}%.")
    print("Quiz Game Completed.")
    print("____________ Thank You for Playing! ____________")
else:
    print("Quiz exited. Come back soon!")