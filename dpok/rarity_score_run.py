import argparse
from dpok_nft.rarity.rarity_engine import RarityScore
import torch
import os
import time
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vit_weights", required=True, help="Path to the pretrained ViT model weights file.")
    parser.add_argument("--generated_images_path", type=str, required=True, help="Paths to the generated images.")
    parser.add_argument("--enable_weighting", action="store_true", help="Enable weighted averaging of scores.")
    parser.add_argument("--enable_mean", action="store_true", help="Enable mean calculation of scores.")
    parser.add_argument("--infer", action="store_true")
    parser.add_argument("--generated_images_path_2", type=str, required=True, help="Paths to the generated images.")
    return parser.parse_args()
    

def main(vit_weights_path, generated_images_path):
    if args.enable_weighting:
        rare_weight = 1
        not_rare_weight = 0
    else:
        rare_weight = 1
        not_rare_weight = 1
        
    if args.infer:
        rarity_model = RarityScore(vit_model_weights_path=vit_weights_path,rare_weight=rare_weight,not_rare_weight=not_rare_weight)
        generated_images_1 = [os.path.join(generated_images_path, filename) 
                              for filename in os.listdir(generated_images_path) 
                              if filename.endswith(('.png'))]
        generated_images_2 = [os.path.join(args.generated_images_path_2, filename) 
                              for filename in os.listdir(args.generated_images_path_2) 
                              if filename.endswith(('.png'))]
        for image_path_1, image_path_2 in zip(generated_images_1, generated_images_2):
            reward_1, image_1 = rarity_model.compute_reward(image_path_1)
            reward_2, image_2 = rarity_model.compute_reward(image_path_2)
            image_1.save("./image1.png")
            image_2.save("./image2.png")
            img_root = image_path_1.split('/')[-1]
            print(f"1: reward: {reward_1}, image_path:{image_path_1.split('/')[-1]}")
            print(f"2: reward: {reward_2}, image_path:{image_path_2.split('/')[-1]}")
            time.sleep(1)
    rarity_model = RarityScore(vit_model_weights_path=vit_weights_path,rare_weight=rare_weight,not_rare_weight=not_rare_weight)
    generated_images = [os.path.join(generated_images_path, filename) 
                              for filename in os.listdir(generated_images_path) 
                              if filename.endswith(('.png'))]
    # print(generated_images)
    for image_path in generated_images:
        
        rarity_model.compute_reward(image_path=image_path)
        # time.sleep(0.1)
    
    # rarity_model.normalize_reward()
    # print(f"reward: {len(rarity_model.rewards >= 0.5)}")
    num_rewards_above_0_5 = len([reward for reward in rarity_model.rewards if reward >= 0.5])
    print(f"num rare: {num_rewards_above_0_5}")
    if args.enable_weighting:
        weighted_rarity_score = rarity_model.calculate_weighted_average()
        print("Weighted Rarity Score: ", weighted_rarity_score)
    elif args.enable_mean:
        mean_rarity_score = rarity_model.calculate_mean_reward()
        print("Mean Rarity Score: ", mean_rarity_score)
    else:
        mean_rarity_score = rarity_model.calculate_mean_reward()
        print("Mean Rarity Score (Default): ", mean_rarity_score)


if __name__ == "__main__":
    args = parse_arguments()
    main(args.vit_weights, args.generated_images_path)

# Example usage:
# python rarity_score_run.py --vit_weights path/to/pretrained/vit/model/weights --generated_images_path path/to/generated/images/folder