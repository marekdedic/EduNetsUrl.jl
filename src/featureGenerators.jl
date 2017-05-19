export ngramFeatureGenerator, unigramFeatureGenerator, bigramFeatureGenerator, trigramFeatureGenerator;

"Generates an array of all the n-grams (substrings of length n) from a given string."
function ngrams(input::AbstractString, n::Int)::Vector{AbstractString}
	output = Vector{AbstractString}(max(length(input) - n + 1, 0));
	indices = Vector{Int}(length(input));
	i = 1;
	j = 1;
	while i <= endof(input)
		indices[j] = i;
		i = nextind(input, i);
		j += 1;
	end
	for i in 1:length(output)
		output[i] = input[indices[i]:indices[i + n - 1]];
	end
	return output;
end

function ngramFeatureGenerator(input::AbstractString, modulo::Int, n::Int; T::DataType = Float32)::Vector{Float32}
	output = zeros(T, modulo);
	for i in ngrams(input, n)
		index = mod(hash(i), modulo);
		output[index + 1] += 1;
	end
	return output;
end

unigramFeatureGenerator(input::AbstractString, modulo::Int; T::DataType = Float32) = ngramFeatureGenerator(input, modulo, 1; T = T);
bigramFeatureGenerator(input::AbstractString, modulo::Int; T::DataType = Float32) = ngramFeatureGenerator(input, modulo, 2; T = T);
trigramFeatureGenerator(input::AbstractString, modulo::Int; T::DataType = Float32) = ngramFeatureGenerator(input, modulo, 3; T = T);
