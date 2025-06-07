from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def create_confusion_matrix_analysis(df):
    """
    Create confusion matrix and metrics for safety classification
    where 'safe' is treated as positive class and 'unsafe' as negative class
    """
    
    # Create predicted labels based on success and original_label
    def get_predicted_label(row):
        if row['success']:
            # Model agreed with original label
            return row['original_label']
        else:
            # Model disagreed with original label
            return 'unsafe' if row['original_label'] == 'safe' else 'safe'
    
    df['predicted_label'] = df.apply(get_predicted_label, axis=1)
    
    # Calculate confusion matrix components
    tp = len(df[(df['original_label'] == 'safe') & (df['success'] == True)])      # True Positive
    tn = len(df[(df['original_label'] == 'unsafe') & (df['success'] == True)])    # True Negative
    fn = len(df[(df['original_label'] == 'safe') & (df['success'] == False)])     # False Negative
    fp = len(df[(df['original_label'] == 'unsafe') & (df['success'] == False)])   # False Positive
    
    print("="*60)
    print("CONFUSION MATRIX ANALYSIS")
    print("="*60)
    print("Class Definition:")
    print("  Positive Class: 'safe'")
    print("  Negative Class: 'unsafe'")
    print()
    
    print("Confusion Matrix Components:")
    print(f"  True Positives (TP):  {tp} - Correctly identified 'safe' content")
    print(f"  True Negatives (TN):  {tn} - Correctly identified 'unsafe' content") 
    print(f"  False Negatives (FN): {fn} - 'Safe' content misclassified as 'unsafe'")
    print(f"  False Positives (FP): {fp} - 'Unsafe' content misclassified as 'safe'")
    print()
    
    # Calculate metrics
    total = tp + tn + fn + fp
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print("Performance Metrics:")
    print(f"  Accuracy:    {accuracy:.4f} ({accuracy:.2%})")
    print(f"  Precision:   {precision:.4f} ({precision:.2%}) - Of predicted 'safe', how many were actually safe")
    print(f"  Recall:      {recall:.4f} ({recall:.2%}) - Of actual 'safe', how many were correctly identified")
    print(f"  Specificity: {specificity:.4f} ({specificity:.2%}) - Of actual 'unsafe', how many were correctly identified")
    print(f"  F1-Score:    {f1_score:.4f}")
    print()
    
    # Label distribution
    print("Label Distribution:")
    label_counts = df['original_label'].value_counts()
    for label, count in label_counts.items():
        percentage = count / len(df) * 100
        print(f"  {label}: {count} samples ({percentage:.1f}%)")
    print()
    
    # Create confusion matrix using sklearn for verification
    y_true = df['original_label'].map({'safe': 1, 'unsafe': 0})  # safe=1 (positive), unsafe=0 (negative)
    y_pred = df['predicted_label'].map({'safe': 1, 'unsafe': 0})
    
    cm = confusion_matrix(y_true, y_pred)
    
    print("Confusion Matrix (sklearn verification):")
    print("                 Predicted")
    print("                Unsafe  Safe")
    print(f"Actual Unsafe    {cm[0,0]:4d}   {cm[0,1]:4d}")
    print(f"       Safe      {cm[1,0]:4d}   {cm[1,1]:4d}")
    print()
    
    # Detailed breakdown
    print("Detailed Breakdown:")
    breakdown = df.groupby(['original_label', 'success']).size().unstack(fill_value=0)
    if 'success' in breakdown.columns:
        breakdown.columns = ['Incorrect', 'Correct']
    print(breakdown)
    print()
    
    # Classification report
    print("Classification Report:")
    target_names = ['unsafe', 'safe']  # Order matters for sklearn
    print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))
    
    return {
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'accuracy': accuracy, 'precision': precision, 'recall': recall,
        'specificity': specificity, 'f1_score': f1_score,
        'confusion_matrix': cm
    }

def plot_confusion_matrix(df, save_path=None):
    """Create a visual confusion matrix using sklearn's ConfusionMatrixDisplay"""
    # Create predicted labels
    def get_predicted_label(row):
        if row['success']:
            return row['original_label']
        else:
            return 'unsafe' if row['original_label'] == 'safe' else 'safe'
    
    df['predicted_label'] = df.apply(get_predicted_label, axis=1)
    
    # Create confusion matrix
    y_true = df['original_label']
    y_pred = df['predicted_label']
    
    labels = ['unsafe', 'safe']
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Create the display
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    ax.set_title('Confusion Matrix\n(Safe as Positive Class)', fontsize=14, pad=20)
    
    # Improve layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    
    return cm, disp
