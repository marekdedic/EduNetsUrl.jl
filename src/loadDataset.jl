import GZip;
import JSON;

export loadUrl, loadThreatGrid;

function loadUrlsFromFile(file::AbstractString)::Vector{AbstractString}
	output = Vector{AbstractString}(0);
	GZip.open(file) do g
		for line in eachline(g)
			json = JSON.parse(line);
			try
				ohttp = json["ohttp"];						
				if ohttp != nothing
					url = ohttp["Host"] * ohttp["uri"];
					push!(output, url);
				end
			end
		end
	end
	return output;
end							

function loadUrl(file::AbstractString; batchSize::Int = 6000, featureCount::Int = 2053, featureGenerator::Function = trigramFeatureGenerator, parallel::Bool = false, T::DataType = Float32)::IterableParser
	open(file, "r") do fid
		table = readcsv(fid);
	end

	if any(table[:, 3] .!= "legit")
		table = table[table[:, 3] .!= "legit", :];
	end

	urls = table[:, 1];
	labels = (table[:, 3] .!= "legit") + 1;
	return IterableParser(convert(Vector{AbstractString}, urls), convert(Vector{Int}, labels), batchSize; featureCount = featureCount, featureGenerator = featureGenerator, parallel = parallel, T = T);
end


function loadThreatGrid(dir::AbstractString; labeller::Function = countingLabeller, batchSize::Int = 6000, featureCount::Int = 2053, featureGenerator::Function = trigramFeatureGenerator, parallel::Bool = false, T::DataType = Float32)::IterableParser
	urls = Vector{Vector{AbstractString}}(0);
	labels = Vector{Int}(0);
	for (root, dirs, files) in walkdir(dir)
		for file in filter(x-> ismatch(r"\.joy\.json\.gz$", x), files)
			path = joinpath(root, file);
			filename = replace(path, r"^(.*)\.joy\.json\.gz$", s"\1");
			if isfile(filename * ".vt.json")
				push!(urls, loadUrlsFromFile(filename * ".joy.json.gz"));
				push!(labels, labeller(filename * ".vt.json"));
			end
		end
	end
	return IterableParser(vcat(urls...), labels, batchSize; featureCount = featureCount, featureGenerator = featureGenerator, parallel, T = T)
end
