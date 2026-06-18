import os
import json
import glob
import tqdm
import torch.distributed as dist

def run_evaluation_pred(args, engine, rank, world_size):
    all_items = []
    if os.path.isdir(args.json_path):
        json_files = sorted(glob.glob(os.path.join(args.json_path, "*.json")))
        for jf in json_files:
            try:
                with open(jf, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    all_items.extend(data if isinstance(data, list) else [data])
            except Exception as e:
                print(f"读取 JSON 失败 {jf}: {e}")
    else:
        with open(args.json_path, 'r', encoding='utf-8') as f:
            all_items = json.load(f)

    model_tag = os.path.basename(args.model_name.strip('/'))
    json_tag = os.path.basename(args.json_path.strip('/'))
    
    existing_tmp_files = glob.glob(os.path.join(args.output_dir, f"tmp_{model_tag}_{json_tag}_rank_*.json"))
    
    global_processed_ids = set()
    for tmp_f in existing_tmp_files:
        try:
            with open(tmp_f, 'r', encoding='utf-8') as f:
                tmp_data = json.load(f)
                for res in tmp_data:
                    global_processed_ids.add(res['id'])
        except:
            continue

    if rank == 0:
        print(f"📊 全局统计: 总任务 {len(all_items)}, 已完成 {len(global_processed_ids)}, 剩余 {len(all_items) - len(global_processed_ids)}")

    remaining_items = [item for item in all_items if item['id'] not in global_processed_ids]

    my_items = remaining_items[rank::world_size] 

    output_res_path = os.path.join(args.output_dir, f"tmp_{model_tag}_{json_tag}_rank_{rank}.json")

    results = []
    if os.path.exists(output_res_path):
        try:
            with open(output_res_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
        except:
            results = []

    for item in tqdm.tqdm(my_items, desc=f"Rank {rank}"):
        video_path = os.path.join(args.video_dir, item['video_name'])
        gt = item.get('answer', 'Unknown')
        
        try:
            raw_response = engine.call(video_path, item)
            
            from utils import extract_answer 
            pred = extract_answer(raw_response)
            
            gt_str = str(gt).strip()
            pred_str = str(pred).strip()

            res_item = {
                "id": item['id'],
                "video_name": item['video_name'],
                "ground_truth": gt_str,
                "prediction": raw_response,
                "extracted_pred": pred_str,
                "is_correct": (pred_str == gt_str) 
            }
            results.append(res_item)

            with open(output_res_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4, ensure_ascii=False)
                
        except Exception as e:
            print(f"Rank {rank} 处理 {item['id']} 失败: {e}")
            continue

    if dist.is_initialized():
        dist.barrier()
    
def merge_results_and_report_pred(args):
    current_rank = getattr(args, 'rank', 0)
    if current_rank != 0:
        return

    model_tag = os.path.basename(args.model_name.strip('/'))
    json_tag = os.path.basename(args.json_path.strip('/'))
    
    file_pattern = os.path.join(args.output_dir, f"tmp_{model_tag}_{json_tag}_rank_*.json")
    result_files = glob.glob(file_pattern)
    
    all_results = []
    for r_file in sorted(result_files):
        try:
            with open(r_file, 'r', encoding='utf-8') as f:
                all_results.extend(json.load(f))
        except Exception as e:
            print(f"读取文件 {r_file} 出错: {e}")

    if not all_results:
        print("❌ 未找到任何结果文件。")
        return

    total = len(all_results)
    na_results = [x for x in all_results if x.get('extracted_pred') == "N/A"]
    na_count = len(na_results)
    
    valid_results = [x for x in all_results if x.get('extracted_pred') != "N/A"]
    valid_total = len(valid_results)
    correct = sum(1 for x in valid_results if x.get('is_correct'))
    
    overall_acc = (sum(1 for x in all_results if x.get('is_correct')) / total * 100) if total > 0 else 0
    valid_acc = (correct / valid_total * 100) if valid_total > 0 else 0

    final_output_data = {
        "summary": {
            "model_name": args.model_name,
            "dataset_json": args.json_path,
            "total_samples": total,
            "na_count": na_count,
            "valid_samples": valid_total,
            "correct_count": correct,
            "overall_accuracy": f"{overall_acc:.2f}%",
            "valid_accuracy": f"{valid_acc:.2f}%"
        },
        "results": all_results
    }

    print("-" * 50)
    print(f"📊 最终汇总报告 [{json_tag}]")
    print("-" * 50)
    for key, value in final_output_data["summary"].items():
        print(f"{key:20}: {value}")
    print("-" * 50)

    final_output_path = os.path.join(args.output_dir, f"final_{model_tag}_{json_tag}_results.json")
    with open(final_output_path, 'w', encoding='utf-8') as f:
        json.dump(final_output_data, f, indent=4, ensure_ascii=False)
    
    print(f"💾 包含统计信息的汇总文件已保存至: {final_output_path}\n")
