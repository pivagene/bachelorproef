# Step 1: Read the CSV file and write to a FASTA file and choose how to make the db
csv_file = 'AMP_species_list_COX1.csv'
df = pd.read_csv(csv_file)
# method 1: alle fasta bestanden uit csv halen om db te maken 
with open('COX1_db.fasta', 'w') as fasta_file:   # alle fasta bestanden uit csv halen om db te maken
    for index, row in df.iterrows():
        fasta_file.write(f">{row['FASTA']}\n")
# method 2: 1 fasta bestand uit csv halen om db te maken
# Extract the value from the 'fasta' column at row 604
fasta_value = df.at[604, 'FASTA']

# Save the extracted value as a FASTA file
with open('COX1_db.fasta', 'w') as fasta_file:
    fasta_file.write(f'>sequence_604\n{fasta_value}\n')
# voor deze stap moet je blast+ installeren en in path zetten bij de omgevingsvariabelen (https://blast.ncbi.nlm.nih.gov/Blast.cgi?CMD=Web&PAGE_TYPE=BlastDocs&DOC_TYPE=Download)
# Step 2: Create the BLAST database (hier verschillende groottes gebruiken kan ook via ncbi gdn worden expected time is 5 uur tho)
subprocess.run(['makeblastdb', '-in', 'COX1_db.fasta', '-dbtype', 'nucl', '-out', 'COX1_blastdb'])



# Function to perform local BLAST alignment and extract alignment score and bit score
def get_blast_scores(sequence, db='COX1_blastdb'):
    # Use a unique filename for each thread
    temp_fasta_filename = f'temp_query_{threading.get_ident()}.fasta'
    
    # Write the sequence to a temporary FASTA file
    with open(temp_fasta_filename, 'w') as temp_fasta:
        temp_fasta.write(f'>query\n{sequence}\n')
    
    # Run the local BLAST search
    result = subprocess.run(
        ['blastn', '-query', temp_fasta_filename, '-db', db, '-outfmt', '5'],
        capture_output=True, text=True
    )
    
    # Remove the temporary FASTA file
    os.remove(temp_fasta_filename)
    
    # Check if the BLAST search produced valid output
    if result.stdout.strip():
        # Parse the BLAST XML output
        blast_records = NCBIXML.parse(io.StringIO(result.stdout))
        for blast_record in blast_records:
            if blast_record.alignments:
                hsp = blast_record.alignments[0].hsps[0]
                return hsp.score, hsp.bits
    return 0, 0



#stond in for lus om sequences om te zetten in scores
# blast_score, bit_score = get_blast_scores(sequence) dit moet met fasta bestanden gedaan worden
#blast_scores.append(blast_score)
    #bit_scores.append(bit_score)


# Add BLAST scores and bit scores to the features DataFrame
#features_df['blast_score'] = blast_scores
#features_df['bit_score'] = bit_scores
