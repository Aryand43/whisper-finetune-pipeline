from model_aggregation import average_checkpoints

model_paths = [
    "checkpoints/hearty-bee-7.pt",      
    "checkpoints/smart-smoke-5.pt",      
    "checkpoints/eager-haze-6.pt",       
    "checkpoints/wandering-river-4.pt"   
]

weights = [0.25, 0.25, 0.25, 0.25]

average_checkpoints(model_paths, weights, save_path="aggregated_model.bin")
