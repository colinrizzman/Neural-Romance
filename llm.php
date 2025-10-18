<?php

/*
    github.com/colinrizzman

    -----

    This uses llama.cpp (https://github.com/ggml-org/llama.cpp) to query an LLM model to generate training data.
    
    LLM Model:  Qwen3-30B-A3B-Instruct-2507-Q4_K_M
    Model Card: https://huggingface.co/lmstudio-community/Qwen3-30B-A3B-Instruct-2507-GGUF
    Download:   https://huggingface.co/lmstudio-community/Qwen3-30B-A3B-Instruct-2507-GGUF/blob/main/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf

    Potential improvements:
    - The random numbers for the next prompt could be created while waiting for the current prompt to be generated.

    Prerequisits:
    - apt install php-cli php-curl
    
    -----

    Vulkan0: Tesla P40 (24576 MiB, 6191 MiB free)
    Vulkan1: AMD Instinct MI60 / MI50 (RADV VEGA20) (32752 MiB, 32731 MiB free)
    Vulkan2: AMD Radeon RX 6300 (RADV NAVI24) (2032 MiB, 1216 MiB free)
    Vulkan3: AMD Instinct MI60 / MI50 (RADV VEGA20) (32752 MiB, 32731 MiB free)

    llama-server --port 8081 --device Vulkan3 --threads 66 --mlock --ctx-size 512 --batch-size 512 --n-gpu-layers -1 -m Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf

    --device Vulkan1,Vulkan3
    --no-kv-offload
    --no-mmap
    --cpu-strict 1
    
*/

if($argc < 2){die("Usage: php q.php <port>\n");}
$port = intval($argv[1]);
$inputs = array();

function rndPrompt()
{
    global $inputs;
    for($i = 0; $i <= 26; $i++){$inputs[$i] = random_int(0, 9);}
    $prompt  = "these attributes describe me\n\n";
    $prompt .= "fatness: " . $inputs[0];
    $prompt .= "curiosity: " . $inputs[1];
    $prompt .= "empathy: " . $inputs[2];
    $prompt .= "ambition: " . $inputs[3];
    $prompt .= "positivity: " . $inputs[4];
    $prompt .= "depressive: " . $inputs[5];
    $prompt .= "creativity: " . $inputs[6];
    $prompt .= "intellectual: " . $inputs[7];
    $prompt .= "spiritual: " . $inputs[8];
    $prompt .= "traditional: " . $inputs[9];
    $prompt .= "loyalty: " . $inputs[10];
    $prompt .= "stability: " . $inputs[11];
    $prompt .= "emotionality: " . $inputs[12];
    $prompt .= "nurturing: " . $inputs[13];
    $prompt .= "affectionate: " . $inputs[14];
    $prompt .= "possessive: " . $inputs[15];
    $prompt .= "dominant: " . $inputs[16];
    $prompt .= "openness: " . $inputs[17];
    $prompt .= "defiant: " . $inputs[18];
    $prompt .= "independent: " . $inputs[19];
    $prompt .= "trustworthy: " . $inputs[20];
    $prompt .= "sociability: " . $inputs[21];
    $prompt .= "humor: " . $inputs[22];
    $prompt .= "risk-taking: " . $inputs[23];
    $prompt .= "adventurous: " . $inputs[24];
    $prompt .= "quirkiness: " . $inputs[25];
    $prompt .= "crazy: " . $inputs[26];
    $prompt .= "\n\nbased on these attributes give me a percentage of how likely i am to find love, give me only the percentage";
    return "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n" . $prompt . "<|im_end|>\n<|im_start|>assistant\n";
}

/*
    prompt	            Text prompt to complete
    n_predict	        Max tokens to generate
    temperature	        Sampling temperature
    top_p	            Nucleus sampling (probability mass cutoff)
    top_k	            Top-K token sampling
    repeat_penalty	    Penalize recently seen tokens
    presence_penalty	Similar to OpenAI penalties
    seed	            RNG seed (per-request override)
    stop	            Array of strings that stop generation
    n_probs	            If >0, returns top-N token probabilities per step
    grammar	            Grammar-based constraint (structured output)
    cache_prompt	    Whether to reuse KV cache for same prefix
    stream	            Stream output (true for SSE-like streaming)
    penalize_nl	        Whether newlines are penalized in repeat penalty

    tfs_z, typical_p, min_p	                Alternative sampling filters
    mirostat, mirostat_tau, mirostat_eta	Mirostat adaptive sampling (0 = off)
*/
$payload = [
    "prompt" => rndPrompt(),
    "top_k" => 20,
    "top_p" => 0.8,
    "min_p" => 0,
#    "cache_prompt" => true,
    "n_predict" => 3,
    "temperature" => 0.7,
    "seed" => random_int(0, PHP_INT_MAX),
];

$ch = curl_init();
$options = [
    CURLOPT_URL => "http://127.0.0.1:" . $port . "/completion",
    CURLOPT_RETURNTRANSFER => true,
    CURLOPT_POST => true,
    CURLOPT_HTTPHEADER => ["Content-Type: application/json"],
    CURLOPT_POSTFIELDS => json_encode($payload),
];
curl_setopt_array($ch, $options);
while(1)
{
    $payload['prompt'] = rndPrompt();
    $payload['seed'] = random_int(0, PHP_INT_MAX);
    curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($payload));
    $response = curl_exec($ch);
    if($response === false){echo "curl error: " . curl_error($ch) . "\n";}else
    {
        if(!isset(json_decode($response)->content)){echo "error: no content\n";continue;}
        $out = str_replace("\n", "", json_decode($response)->content);
        if(strlen($out) > 2 && $out[2] != '%'){echo "failed: " . $out . "\n";continue;}
        $output = "";
        for($i = 0; $i <= 26; $i++){$output .= number_format(floatval($inputs[$i])/9, 2) . " ";}
        $output .= number_format(floatval(str_replace('%', '', $out))/100, 2);
        file_put_contents("training_data.txt", $output . "\n", LOCK_EX | FILE_APPEND);
        echo "success: " . $out . "\n";
    }
}
curl_close($ch);
