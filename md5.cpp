#include <iostream>
#include <iomanip>
#include <sstream>
#include "mpi.h"
#include <chrono>
#include "openssl/md5.h"

std::vector<std::string> splitString(const std::string& inputString) {
    std::vector<std::string> substrings;
    size_t length = inputString.length();

    for (size_t i = 0; i < length; i += length / 7) {
        substrings.push_back(inputString.substr(i, length / 7));
    }

    return substrings;
}
std::string md5(const std::string& input) {
    MD5_CTX context;
    MD5_Init(&context);
    MD5_Update(&context, input.c_str(), input.length());

    unsigned char result[MD5_DIGEST_LENGTH];
    MD5_Final(result, &context);

    std::stringstream md5Stream;
    for (int i = 0; i < MD5_DIGEST_LENGTH; ++i) {
        md5Stream << std::hex << std::setw(2) << std::setfill('0') << (int)result[i];
    }

    return md5Stream.str();
}
std::string get_next_password(std::string current_password, std::string alphabet, int letter_num, int num) {
    current_password[letter_num] = alphabet[num];
    return current_password;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    std::string alphabet = "QWERTYUIOPASDFGHJKLZXCVBNMqwertyuiopasdfghjklzxcvbnm1234567890";
    std::vector<std::string> chunks = splitString(alphabet);

    int rank_id, num_proc;
    int finished = 0;
    const int alphabet_len = alphabet.length() - 1;

    MPI_Status status;
    MPI_Request request;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_id);
    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);

//    MPI_Bcast(&finished, 1, MPI_INT, 0, MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    std::string init = "QQQQQ";
    std::string word = "vA7ny";//"W43xs";
    std::string word_hash = md5(word);
    for (uint i = 0; finished != 1; i++) {
//        init = get_next_password(init, i % 5 == 0 ? chunks[rank_id] : alphabet, i % 5, i % 5 == 0 ? i % (chunks[rank_id].length() - 1) : i % alphabet_len);
////        std::cout << "Runk: " << rank_id << " " << init << '\n';
//        if (md5(init) == word_hash) {
//            finished = 1;
//            auto end = std::chrono::high_resolution_clock::now();
//            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
//            std::cout << "Runk: " << rank_id << " found the word:" << init << " for time: " << duration <<'\n';
//            MPI_Abort(MPI_COMM_WORLD, 0);
//        }
//        MPI_Bcast(&finished, 1, MPI_INT, 0, MPI_COMM_WORLD);
        for (int frst = 0; frst < chunks[rank_id].length() - 1; frst++) {
            for (int scnd = 0; scnd < alphabet_len; scnd++) {
                for (int thrd = 0; thrd < alphabet_len; thrd++) {
                    for (int fth = 0; fth < alphabet_len; fth++) {
                        for (int ffth = 0; ffth < alphabet_len; ffth++) {
                            init[0] = chunks[rank_id][frst];
                            init[1] = alphabet[scnd];
                            init[2] = alphabet[thrd];
                            init[3] = alphabet[fth];
                            init[4] = alphabet[ffth];
                            if (md5(init) == word_hash) {
                                finished = 1;
                                auto end = std::chrono::high_resolution_clock::now();
                                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
                                std::cout << "Runk: " << rank_id << " found the word: " << init << " for time: " << duration <<'\n';
                                MPI_Abort(MPI_COMM_WORLD, 0);
                            }
                        }
                    }
                }
            }
        }
    }
    std::cout << "Runk: " << rank_id << " finished: " << finished << " word: " << word << "\n";

    MPI_Finalize();
    return 0;
}