<?php
header("Access-Control-Allow-Origin: *");
header("Access-Control-Allow-Headers: Content-Type");
header("Access-Control-Allow-Methods: POST, OPTIONS");
header("Content-Type: application/json");

// Handle OPTIONS preflight
if ($_SERVER['REQUEST_METHOD'] === 'OPTIONS') {
    http_response_code(200);
    exit;
}

// Only POST allowed
if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    echo json_encode(['error' => 'Only POST allowed']);
    exit;
}


$api_key = "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"; 
$endpoint = "https://openrouter.ai/api/v1/chat/completions";
$model = "nex-agi/deepseek-v3.1-nex-n1:free"; 


$input = json_decode(file_get_contents("php://input"), true);
if (!$input || empty($input['message'])) {
    echo json_encode(['error' => 'Invalid input']);
    exit;
}

$user_message = trim($input['message']);


$data = [
    "model" => $model,
    "messages" => [
        [
            "role" => "user",
            "content" => $user_message
        ]
    ],
    "provider" => [
        "sort" => "throughput"
    ],
    "temperature" => 0.7
];


$ch = curl_init($endpoint);
curl_setopt_array($ch, [
    CURLOPT_RETURNTRANSFER => true,
    CURLOPT_POST => true,
    CURLOPT_POSTFIELDS => json_encode($data),
    CURLOPT_HTTPHEADER => [
        "Content-Type: application/json",
        "Authorization: Bearer $api_key"
    ]
]);

$response = curl_exec($ch);
$http_code = curl_getinfo($ch, CURLINFO_HTTP_CODE);
curl_close($ch);


if ($http_code !== 200) {
    $message = "OpenRouter API error";
    if ($http_code == 429) $message = "Quota exceeded. Please check your plan.";
    if ($http_code == 401) $message = "Invalid API key.";
    echo json_encode([
        'error' => $message,
        'status' => $http_code,
        'response' => $response
    ]);
    exit;
}


$response_data = json_decode($response, true);
$ai_response = $response_data['choices'][0]['message']['content'] ?? 'No response';


echo json_encode(['response' => $ai_response]);
