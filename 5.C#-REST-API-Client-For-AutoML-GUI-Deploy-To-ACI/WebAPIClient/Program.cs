using System;
using System.Text;
using System.Threading.Tasks;
using System.Net.Http;
using System.Net.Http.Headers;

using Newtonsoft.Json;

namespace WebAPIClient
{
    class Program
    {

        private static readonly HttpClient _client = new HttpClient();
        private static readonly String RESTAPI_URL = "<Your ACI URL>.azurecontainer.io/score";
        private static readonly String RESTAPI_KEY = "<Your Key>";

        private static readonly string httpPostBody = "{\"data\": [ {\"age\": 24,\"job\": \"technician\",\"marital\": \"single\",\"education\": \"university.degree\",\"default\": \"no\",\"housing\": \"no\",\"loan\": \"yes\",\"contact\": \"cellular\",\"month\": \"jul\",\"duration\": 109, \"campaign\": 3,\"pdays\": 999,\"previous\": 0,\"poutcome\": \"nonexistent\",\"emp.var.rate\": 1.4,\"cons.price.idx\": 93.918,\"cons.conf.idx\": -42.7,\"euribor3m\": 4.963,\"nr.employed\": 5228.1}]}";

        static async Task Main(string[] args)
        {
            await ProcessRepositories();
            Console.ReadLine();
        }

        private static async Task ProcessRepositories()
        {
            // REST API 組み立て
            var request = new HttpRequestMessage
            {
                Method = HttpMethod.Post,
                RequestUri = new Uri(RESTAPI_URL)
            };
            request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", RESTAPI_KEY);
            request.Content = new StringContent(httpPostBody, Encoding.UTF8, "application/json");

            // REST API 呼び出し
            var response = await _client.SendAsync(request);
            var responseBody = response.Content.ReadAsStringAsync().Result;
            string result = "";

            if (response.IsSuccessStatusCode) {
                // 文字列整形
                var responseBodyClearnuped = responseBody.Replace("\"{", "{").Replace("\"}", "}").Replace("}\"", "}").Replace("\\\"","\"");

                // JSON オブジェクト作成
                dynamic responseBodyJOSN  = JsonConvert.DeserializeObject(responseBodyClearnuped);
                result = responseBodyJOSN?.result[0];

            } else {
                result = $"Error: {response.ReasonPhrase} \r {responseBody}"; 
            }

            Console.WriteLine($"result: {result}");
        }
    }
}
