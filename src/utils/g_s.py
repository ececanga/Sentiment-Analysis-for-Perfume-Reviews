import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_slide():
    # 1. Setup Figure (16:9 aspect ratio, high resolution)
    fig, ax = plt.subplots(figsize=(16, 9), dpi=300)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Hide axes
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # ---------------------------------------------------------
    # 2. Header
    # ---------------------------------------------------------
    plt.text(5, 92, "Worked Example Setup (Randomly Generated Artificial Intelligence Perfume Reviews)", 
             fontsize=20, fontweight='bold', color='#2c3e50', ha='left')
    
    # Horizontal line under title
    ax.add_patch(patches.Rectangle((5, 90), 90, 0.3, facecolor='#2c3e50'))

    # ---------------------------------------------------------
    # 3. Left Column (Goal & Input Schema) - approx x=5 to x=45
    # ---------------------------------------------------------
    
    # --- Goal Section ---
    plt.text(5, 82, "🎯 Goal", fontsize=16, fontweight='bold', color='#2980b9')
    plt.text(5, 78, "Demonstrate the complete pipeline end to end on a\nsmall, concrete worked example.", 
             fontsize=14, color='#34495e', va='top', linespacing=1.6)

    # --- Input Schema Section ---
    plt.text(5, 65, "📅 Input Schema", fontsize=16, fontweight='bold', color='#2980b9')
    plt.text(5, 61, "Columns include:\n• Perfume\n• Final Transcript\n• Transcript (Optional)", 
             fontsize=14, color='#34495e', va='top', linespacing=1.6)

    # --- Important Note ---
    # Draw a light grey box for the note
    rect = patches.FancyBboxPatch((5, 42), 40, 12, boxstyle="round,pad=1", 
                                  linewidth=1, edgecolor='#bdc3c7', facecolor='#ecf0f1')
    ax.add_patch(rect)
    
    plt.text(7, 51, "Note:", fontsize=12, fontweight='bold', color='#c0392b')
    plt.text(7, 48, "The file 'YouTube_Perfume_Reviews_FullyFormatted.csv'\nis treated as the immutable reference file that\nall emotion engines read from.", 
             fontsize=12, color='#2c3e50', va='top', linespacing=1.5)

    # ---------------------------------------------------------
    # 4. Right Column (Mini-Table Preview) - approx x=50 to x=95
    # ---------------------------------------------------------
    plt.text(50, 82, "📝 Data Preview (Mini-Table)", fontsize=16, fontweight='bold', color='#2980b9')

    # Table Data
    columns = ["Perfume", "Review ID", "Transcript Snippet"]
    rows = [
        ["Citrus Dawn", "R1", "Bright citrus opening, lemon, bergamot, fresh,\nuplifting, musk, vanilla."],
        ["Citrus Dawn", "R2", "Cheerful, cozy, orange blossom, vanilla,\ncomforting, confident."],
        ["Smoky Noir",  "R3", "Dark smoky leather tar, harsh chemical edge,\noverwhelming, sad, brooding."]
    ]

    # Create Table
    table = plt.table(cellText=rows, colLabels=columns, loc='center', 
                      cellLoc='left', colLoc='left',
                      bbox=[0.50, 0.45, 0.45, 0.3]) # [x, y, width, height]

    # Style the Table
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2) # Adjust row height

    # Colorize headers and rows
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('#bdc3c7')
        cell.set_linewidth(1)
        if row == 0:
            cell.set_facecolor('#2980b9')
            cell.set_text_props(color='white', fontweight='bold')
            cell.set_height(0.1)
        else:
            cell.set_facecolor('white')
            cell.set_text_props(color='#2c3e50')
            cell.set_height(0.1)

    # ---------------------------------------------------------
    # 5. Footer
    # ---------------------------------------------------------
    plt.text(50, 5, "Worked example uses three randomly generated artificial intelligence perfume reviews for two perfumes.", 
             fontsize=10, color='#7f8c8d', ha='center', style='italic')

    # 6. Save
    plt.tight_layout()
    output_filename = "workedExample_slide1.png"
    plt.savefig(output_filename, bbox_inches='tight', pad_inches=0.1)
    print(f"✅ Slide generated successfully: {output_filename}")

# Run the function
if __name__ == "__main__":
    create_slide()