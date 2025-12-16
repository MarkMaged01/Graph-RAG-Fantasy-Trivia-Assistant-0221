import os
import sys

# Ensure we can import the processor
try:
    from input_processing import FPLPreprocessor
except ImportError:
    print("❌ Error: 'input_processing.py' not found in this folder.")
    sys.exit(1)

def run_tests():
    # 1. Initialize the Processor
    print("⏳ Initializing FPL Preprocessor...")
    processor = FPLPreprocessor()
    print("✅ System Ready.\n")

    # 2. Define the Test Set (100 Queries)
    queries = [
        "How many total points did Mohamed Salah get in the 2021-22 season?",
        "Compare Erling Haaland and Harry Kane based on goals scored.",
        "Who are the best defenders from Newcastle this season?",
        "What team does Ollie Watkins play for?",
        "Did Kieran Trippier keep a clean sheet in Gameweek 12 of 2022-23?",
        "Show me the minutes played for Bukayo Saka in GW 3.",
        "I need a recommendation for a midfielder from Man City.",
        "Who has a higher ICT index, Kevin De Bruyne or Bruno Fernandes?",
        "List all metrics for Marcus Rashford in the 2022-23 season.",
        "Is Mohamed Salah considered a midfielder or a forward?",
        "Which Arsenal players had the highest threat in GW 38?",
        "How many red cards did Bruno Fernandes receive in 2021-22?",
        "Give me the assist stats for Harry Kane vs Ollie Watkins.",
        "Recommend a forward who costs less than Haaland but has high influence.",
        "Did Man City keep a clean sheet in Gameweek 15?",
        "check haaland stats gw 1",
        "Who is the best captain option from Liverpool for the upcoming week?",
        "What is the creativity score for Kevin De Bruyne?",
        "Compare the total points of Saka and Rashford for the 2022-23 season.",
        "In which gameweek did Mohamed Salah score the most goals?",
        "Find me a differential player from Aston Villa.",
        "Did Kieran Trippier get any yellow cards in GW 5 of 2022-23?",
        "Who is better on assists, De Bruyne or Salah?",
        "Tell me about Ollie Watkins' recent form.",
        "How many minutes did Erling Haaland play in Gameweek 30?",
        "Top 3 defenders based on clean sheets in 2021-22.",
        "Is Kieran Trippier a good buy for GW 20?",
        "What are the fixtures for Spurs in GW 10?",
        "Compare the influence stats of Bruno Fernandes and Kevin De Bruyne.",
        "Did Harry Kane score in Gameweek 38 of 2021-22?",
        "Who has more total points, Watkins or Rashford?",
        "Suggest a replacement for Mohamed Salah.",
        "What is Bukayo Saka's position?",
        "Show me the ICT index trend for Marcus Rashford.",
        "How many goals did Man Utd score in GW 4?",
        "best budget midfielders 2022-23",
        "Compare the minutes played between Haaland and Kane.",
        "Did Bruno Fernandes get an assist in GW 1?",
        "Who took the most penalties for Man City in 2022-23?",
        "Give me the total points for Kieran Trippier in the 2021-22 season.",
        "Which Spurs player had the highest creativity in GW 9?",
        "Should I bench Ollie Watkins for GW 22?",
        "How many clean sheets did Chelsea get in 2021-22?",
        "Compare Salah vs Saka on threat metrics.",
        "Who is the highest scoring player for Arsenal?",
        "What were Erling Haaland's stats in GW 15 of 2022-23?",
        "Recommend a defender from Newcastle with high assist potential.",
        "Did Kevin De Bruyne play in Gameweek 14?",
        "Show me the yellow cards count for Marcus Rashford.",
        "Who is better value, Trippier or a Man City defender?",
        "What team does Harry Kane play for in 2022-23?",
        "details on bruno fernandes gw 3 2021-22",
        "Who had the most bonus points in GW 25?",
        "Compare the clean sheets of Newcastle and Man City defenders. ",
        "How many goals did Ollie Watkins score against Liverpool?",
        "Is Bukayo Saka on penalties?",
        "Who has a higher xG, Haaland or Kane?",
        "Suggest a captain from Man Utd for GW 16.",
        "What is the price of Mohamed Salah?",
        "Did Kieran Trippier assist in Gameweek 2 of 2022-23?",
        "Compare the creativity of De Bruyne and Saka.",
        "How many minutes did Rashford play in the last 5 gameweeks of 2021-22?",
        "Who are the top 5 players by total points in 2022-23?",
        "Did Erling Haaland get a red card in 2022-23?",
        "Recommend a Chelsea player with good upcoming fixtures.",
        "What is the ICT index for Ollie Watkins?",
        "goals scored by saka gw 10-20",
        "Who is the best budget enabler from Aston Villa?",
        "Compare the heatmaps of Salah and Rashford.",
        "Did Spurs keep a clean sheet in GW 28?",
        "How many assists did Kevin De Bruyne provide in 2021-22?",
        "Is Harry Kane fit for Gameweek 32?",
        "Who scored more points in GW 1, Haaland or Salah?",
        "What position does Bruno Fernandes play?",
        "Give me the stats for the highest scoring defender in 2022-23.",
        "Trippier vs Watkins total points comparison.",
        "How many goals did Arsenal concede in GW 5?",
        "Who is the most transferred in player for GW 12?",
        "Did Marcus Rashford score in Gameweek 7 of 2022-23?",
        "Recommend a goalkeeper from Newcastle.",
        "What is the average minutes played for Erling Haaland?",
        "Compare the threat index of Kane and Haaland.",
        "Who has more yellow cards, Bruno Fernandes or Kieran Trippier?",
        "How many points did Mohamed Salah get in double gameweeks?",
        "Who are the essential players for the 2022-23 season?",
        "Did Ollie Watkins start in GW 30?",
        "Compare the clean sheet potential of Man City vs Liverpool.",
        "What is the ownership percentage of Bukayo Saka?",
        "How many goals did Kevin De Bruyne score from outside the box?",
        "rashford stats 2021-22 summary",
        "Who is the best replacement for Harry Kane?",
        "Did Bukayo Saka get any bonus points in GW 33?",
        "Compare the influence of Salah and De Bruyne in big games.",
        "Which Aston Villa player has the most assists?",
        "How many red cards were shown in Man Utd games in 2022-23?",
        "Is Erling Haaland essential for GW 1?",
        "What is the total points difference between Kane and Haaland?",
        "Recommend a differential captain for GW 38.",
        "Did Bruno Fernandes play 90 minutes in GW 2?",
        "Who was the highest scoring player in Gameweek 21 of 2022-23?"
    ]

    # 3. Run Loop and Write to File
    output_file = "test_results.txt"
    
    with open(output_file, "w", encoding="utf-8") as f:
        print(f"🚀 Processing {len(queries)} queries... (Output saving to {output_file})")
        print("-" * 50)
        
        for i, query in enumerate(queries, 1):
            # A. Process Intent
            intent = processor.classify_intent(query)
            
            # B. Process Entities
            entities = processor.extract_entities(query)
            
            # C. Format Output
            result_str = f"Q{i}: {query}\n"
            result_str += f"   ➤ Intent:   {intent}\n"
            result_str += f"   ➤ Entities: {entities}\n"
            result_str += "-" * 50 + "\n"
            
            # Write to file and print to console (optional: comment out print for speed)
            f.write(result_str)
            print(f"✅ Processed Q{i}")

    print(f"\n🎉 Done! Check '{output_file}' for full results.")

if __name__ == "__main__":
    run_tests()