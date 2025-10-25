import os, sys, torch, argparse, re
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from Energy_Loss_Phenomenon_Demo.utils_hr import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rlhf_model", type=str, default="/mnt/data/users/yuchunmiao/aoss/energy_loss/llama2_7b_miao/llama2_7b_klctl0_actor_sharegpt_step1427_lr5e-7_criticreal_hhtradeoffv1_lr1e-6_bs8_recipe_version1_no_norm_cosine_epoch5_node1_max_length5/actor_latest")
    parser.add_argument("--sft_model", type=str, default="/mnt/data/users/yuchunmiao/models//miao_step1_llama2_7b_sharegpt_lr_bf16_epoch3/step1427")
    parser.add_argument("--dataset_path", type=str, default="./Energy_Loss_Phenomenon_Demo/response_new.jsonl")
    parser.add_argument("--hacking_label", type=str, default="./Energy_Loss_Phenomenon_Demo/hacking_label_new.json")
    parser.add_argument("--gpu", type=str, default="0", help="gpu ids")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    set_random_seed(42)
    token_num = 1024
    extracted_field = "alpaca_farm"
    for tag in ["mlp_energyloss"]:
    # for tag in ["mlp_energyloss", "gate_mlp_energyloss", "up_mlp_energyloss"]:

        logger = logging.getLogger('my_logger')
        logger.setLevel(logging.DEBUG)  # 设置日志级别

        # 清除旧的 FileHandler
        if logger.hasHandlers():
            logger.handlers.clear()

        logger_path = os.path.join(os.path.dirname(args.dataset_path), f'avg_energyloss_{tag}_hish_tokennum{token_num}.log')

        if os.path.exists(logger_path):
            os.remove(logger_path)

        file_handler = logging.FileHandler(logger_path)
        file_handler.setLevel(logging.DEBUG)

        # 创建一个日志格式器并将其添加到处理器
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # 将文件处理器添加到日志记录器
        logger.addHandler(file_handler)

        device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

        rlhf_model_name_or_path = args.rlhf_model
        rlhf_model, rlhf_tokenizer = create_llama_model(rlhf_model_name_or_path, device)

        sft_model_name_or_path = args.sft_model
        sft_model, sft_tokenizer = create_llama_model(sft_model_name_or_path, device)

        hacking_index = []
        normal_index = []
        gpt_label_data = read_json(args.hacking_label)
        for item in gpt_label_data:
            if item["response"] == "Yes":
                hacking_index.append(item['query_id'])
            if item["response"] == "No":
                normal_index.append(item['query_id'])

        hacking_index = list(set(hacking_index))
        normal_index = list(set(normal_index))

        all_datasets = read_jsonl(args.dataset_path)

        savepath = os.path.join(os.path.dirname(args.dataset_path),  f"{extracted_field}_energyloss_hist_last_{tag}_num_{token_num}_llama3_judge.png")
            
        eval_dataset_hacking = [item for item in all_datasets if item['index'] in hacking_index]
        eval_dataset_normal = [item for item in all_datasets if item['index'] in normal_index]
    

        avg_rlhf_list, avg_sft_list = [], []
        hacking_list, normal_list = [], []

        for step, batch in enumerate(eval_dataset_hacking):
            index = batch["index"]
            print(f"Begin to process {step}/{len(eval_dataset_hacking)} hacking samples with index {index}")
            logger.info(f"Begin to process {step}/{len(eval_dataset_hacking)} hacking samples with index {index}")
            rlhf_result = evaluation_energyloss_last_layer(rlhf_model, rlhf_tokenizer, batch, tau=0.01, response_tag = 'actor_latest_response', num=token_num, tag=tag)
            sft_result = evaluation_energyloss_last_layer(sft_model, sft_tokenizer, batch, tau=0.01, response_tag = 'actor_num0_response', num=token_num, tag=tag)

            if rlhf_result == None or sft_result == None:
                continue

            avg_rlhf_list.append(torch.mean(rlhf_result).item())
            avg_sft_list.append(torch.mean(sft_result).item())
            hacking_list.append(torch.mean(rlhf_result).item())
            print("sft---{}, rlhf---{}".format(torch.mean(sft_result).item(), torch.mean(rlhf_result).item()))
            logger.info("sft---{}, rlhf---{}".format(torch.mean(sft_result).item(), torch.mean(rlhf_result).item()))

        for step, batch in enumerate(eval_dataset_normal):
            index = batch["index"]
            print(f"Begin to process {step}/{len(eval_dataset_normal)} normal samples with index {index}")
            logger.info(f"Begin to process {step}/{len(eval_dataset_normal)} normal samples with index {index}")
            
            rlhf_result = evaluation_energyloss_last_layer(rlhf_model, rlhf_tokenizer, batch, tau=0.01, response_tag = 'actor_latest_response', num=token_num, tag=tag)
            sft_result = evaluation_energyloss_last_layer(sft_model, sft_tokenizer, batch, tau=0.01, response_tag = 'actor_num0_response', num=token_num, tag=tag)

            if rlhf_result == None or sft_result == None:
                continue

            avg_rlhf_list.append(torch.mean(rlhf_result).item())
            avg_sft_list.append(torch.mean(sft_result).item())
            normal_list.append(torch.mean(rlhf_result).item())
            print("sft---{}, rlhf---{}".format(torch.mean(sft_result).item(), torch.mean(rlhf_result).item()))
            logger.info("sft---{}, rlhf---{}".format(torch.mean(sft_result).item(), torch.mean(rlhf_result).item()))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

        ax1.hist(avg_sft_list, bins=50, alpha=0.5, label='SFT Model', color='blue')
        ax1.hist(avg_rlhf_list, bins=50, alpha=0.5, label='RLHF Model', color='green')
        ax1.set_title('{}: SFT({})-RLHF({})'.format(extracted_field, sum(avg_sft_list)/len(avg_sft_list), sum(avg_rlhf_list)/len(avg_rlhf_list)))
        ax1.set_xlabel('Mean Energyloss Value')
        ax1.set_ylabel('Frequency')
        ax1.legend()

        ax2.hist(hacking_list, bins=50, alpha=0.5, label='Hacking Sample', color='mediumpurple')
        ax2.hist(normal_list, bins=50, alpha=0.5, label='Normal Sample', color='gold')
        avg_hacking_list = sum(hacking_list)/len(hacking_list) if len(hacking_list) > 0 else 0
        avg_normal_list = sum(normal_list)/len(normal_list) if len(normal_list) > 0 else 0
        ax2.set_title('{}: Hacking({})-Normal({})'.format(extracted_field, avg_hacking_list, avg_normal_list))
        ax2.set_xlabel('Mean Energyloss Value')
        ax2.set_ylabel('Frequency')
        ax2.legend()


        fig.suptitle(f'target layer: {tag}')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath, format='png', bbox_inches='tight', pad_inches=0, facecolor='white')

        plt.close('all')

if __name__ == "__main__":
    main()